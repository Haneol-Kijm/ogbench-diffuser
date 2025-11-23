from typing import Any

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import (
    MLP,
    FlowActor,
    GCActor,
    GCDiscreteActor,
    GCValue,
    Identity,
    LengthNormalize,
)


class HIQLFMAgent(flax.struct.PyTreeNode):
    """Hierarchical implicit Q-learning with Flow Matching (FM-HIQL) agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def value_loss(self, batch, grad_params):
        """Compute the IVL value loss.

        This value loss is similar to the original IQL value loss, but involves additional tricks to stabilize training.
        For example, when computing the expectile loss, we separate the advantage part (which is used to compute the
        weight) and the difference part (which is used to compute the loss), where we use the target value function to
        compute the former and the current value function to compute the latter. This is similar to how double DQN
        mitigates overestimation bias.
        """
        (next_v1_t, next_v2_t) = self.network.select("target_value")(
            batch["next_observations"], batch["value_goals"]
        )
        next_v_t = jnp.minimum(next_v1_t, next_v2_t)
        q = batch["rewards"] + self.config["discount"] * batch["masks"] * next_v_t

        (v1_t, v2_t) = self.network.select("target_value")(
            batch["observations"], batch["value_goals"]
        )
        v_t = (v1_t + v2_t) / 2
        adv = q - v_t

        q1 = batch["rewards"] + self.config["discount"] * batch["masks"] * next_v1_t
        q2 = batch["rewards"] + self.config["discount"] * batch["masks"] * next_v2_t
        (v1, v2) = self.network.select("value")(
            batch["observations"], batch["value_goals"], params=grad_params
        )
        v = (v1 + v2) / 2

        value_loss1 = self.expectile_loss(adv, q1 - v1, self.config["expectile"]).mean()
        value_loss2 = self.expectile_loss(adv, q2 - v2, self.config["expectile"]).mean()
        value_loss = value_loss1 + value_loss2

        return value_loss, {
            "value_loss": value_loss,
            "v_mean": v.mean(),
            "v_max": v.max(),
            "v_min": v.min(),
        }

    def low_actor_loss(self, batch, grad_params):
        """Compute the low-level actor loss."""
        v1, v2 = self.network.select("value")(
            batch["observations"], batch["low_actor_goals"]
        )
        nv1, nv2 = self.network.select("value")(
            batch["next_observations"], batch["low_actor_goals"]
        )
        v = (v1 + v2) / 2
        nv = (nv1 + nv2) / 2
        adv = nv - v

        exp_a = jnp.exp(adv * self.config["low_alpha"])
        exp_a = jnp.minimum(exp_a, 100.0)

        # Compute the goal representations of the subgoals.
        goal_reps = self.network.select("goal_rep")(
            jnp.concatenate([batch["observations"], batch["low_actor_goals"]], axis=-1),
            params=grad_params,
        )
        if not self.config["low_actor_rep_grad"]:
            # Stop gradients through the goal representations.
            goal_reps = jax.lax.stop_gradient(goal_reps)
        dist = self.network.select("low_actor")(
            batch["observations"], goal_reps, goal_encoded=True, params=grad_params
        )
        log_prob = dist.log_prob(batch["actions"])

        actor_loss = -(exp_a * log_prob).mean()

        actor_info = {
            "actor_loss": actor_loss,
            "adv": adv.mean(),
            "bc_log_prob": log_prob.mean(),
        }
        if not self.config["discrete"]:
            actor_info.update(
                {
                    "mse": jnp.mean((dist.mode() - batch["actions"]) ** 2),
                    "std": jnp.mean(dist.scale_diag),
                }
            )

        return actor_loss, actor_info

    def high_actor_loss(self, batch, grad_params):
        """Compute the high-level actor loss using Weighted Conditional Flow Matching."""
        # 1. Compute Advantage for Weighting
        v1, v2 = self.network.select("value")(
            batch["observations"], batch["high_actor_goals"]
        )
        nv1, nv2 = self.network.select("value")(
            batch["high_actor_targets"], batch["high_actor_goals"]
        )
        v = (v1 + v2) / 2
        nv = (nv1 + nv2) / 2
        adv = nv - v

        # AWR Weighting
        exp_a = jnp.exp(adv * self.config["high_alpha"])
        exp_a = jnp.minimum(exp_a, 100.0)

        # 2. Prepare Flow Matching Inputs
        # Target Subgoal (x_1)
        target_subgoal = self.network.select("goal_rep")(
            jnp.concatenate(
                [batch["observations"], batch["high_actor_targets"]], axis=-1
            )
        )

        # Sample Noise (x_0)
        rng = self.rng
        rng, key_t, key_n = jax.random.split(rng, 3)
        x_0 = jax.random.normal(key_n, target_subgoal.shape)

        # Sample Time (t)
        t = jax.random.uniform(key_t, (target_subgoal.shape[0], 1))

        # Interpolate (Optimal Transport Path)
        # x_t = (1 - t) * x_0 + t * x_1
        x_t = (1 - t) * x_0 + t * target_subgoal

        # Target Velocity (u_t)
        # u_t = x_1 - x_0
        u_t = target_subgoal - x_0

        # 3. Predict Velocity
        # Condition: [State, Goal]
        # Note: HIQL passes 'high_actor_goals' which are the final goals G
        # We need to encode G first if we want to condition on it properly,
        # but FlowActor expects a concatenated condition vector.
        # Let's check how HIQL does it.
        # HIQL: dist = high_actor(obs, high_actor_goals)
        # GCActor handles concatenation internally.
        # Our FlowActor expects 'condition'.
        # We should concatenate obs and goal here.

        # Assuming goals are raw observations (not encoded yet)
        # We might need to encode them if using visual encoder, but for state-based:
        condition = jnp.concatenate(
            [batch["observations"], batch["high_actor_goals"]], axis=-1
        )

        v_pred = self.network.select("high_actor")(
            x_t, t, condition, params=grad_params
        )

        # 4. Compute Weighted MSE Loss
        # Loss = weight * || v_pred - u_t ||^2
        loss = jnp.mean(exp_a * jnp.square(v_pred - u_t))

        # Additional Monitoring Metrics
        v_pred_norm = jnp.linalg.norm(v_pred, axis=-1).mean()
        u_t_norm = jnp.linalg.norm(u_t, axis=-1).mean()
        weights_mean = exp_a.mean()
        weights_max = exp_a.max()

        # Cosine Similarity
        v_pred_flat = v_pred.reshape(v_pred.shape[0], -1)
        u_t_flat = u_t.reshape(u_t.shape[0], -1)
        cos_sim = jnp.sum(v_pred_flat * u_t_flat, axis=-1) / (
            jnp.linalg.norm(v_pred_flat, axis=-1) * jnp.linalg.norm(u_t_flat, axis=-1)
            + 1e-6
        )

        return loss, {
            "actor_loss": loss,
            "adv": adv.mean(),
            "flow_mse": jnp.mean(jnp.square(v_pred - u_t)),
            "weight/mean": weights_mean,
            "weight/max": weights_max,
            "velocity/pred_norm": v_pred_norm,
            "velocity/target_norm": u_t_norm,
            "velocity/cos_sim": cos_sim.mean(),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f"value/{k}"] = v

        low_actor_loss, low_actor_info = self.low_actor_loss(batch, grad_params)
        for k, v in low_actor_info.items():
            info[f"low_actor/{k}"] = v

        # Update RNG for flow matching sampling
        if rng is not None:
            self = self.replace(rng=rng)

        high_actor_loss, high_actor_info = self.high_actor_loss(batch, grad_params)
        for k, v in high_actor_info.items():
            info[f"high_actor/{k}"] = v

        loss = value_loss + low_actor_loss + high_actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config["tau"] + tp * (1 - self.config["tau"]),
            self.network.params[f"modules_{module_name}"],
            self.network.params[f"modules_target_{module_name}"],
        )
        network.params[f"modules_target_{module_name}"] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, "value")

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the actor.

        It first queries the high-level flow actor to obtain subgoal representations, and then queries the low-level actor
        to obtain raw actions.
        """
        high_seed, low_seed = jax.random.split(seed)

        # --- High-Level Flow Sampling (Euler ODE) ---
        batch_size = observations.shape[0]
        rep_dim = self.config["rep_dim"]

        # 1. Start from Noise x_0
        x = jax.random.normal(high_seed, (batch_size, rep_dim))

        # 2. Condition
        condition = jnp.concatenate([observations, goals], axis=-1)

        # 3. Euler Loop
        steps = 10  # TODO: Hardcoded for now, can be in config
        dt = 1.0 / steps

        def scan_fn(carry, t):
            x = carry
            # Broadcast t to batch
            t_batch = jnp.full((batch_size, 1), t)

            v = self.network.select("high_actor")(x, t_batch, condition)
            x_next = x + v * dt
            return x_next, None

        # Scan over time steps 0, dt, 2dt, ...
        # Note: We need t values.
        t_values = jnp.linspace(0, 1.0, steps, endpoint=False)

        goal_reps, _ = jax.lax.scan(scan_fn, x, t_values)

        # Clip final output (Subgoals are normalized/bounded usually, but Flow can go unbounded.
        # HIQL uses LengthNormalize for targets. We should probably normalize or clip.)
        # For now, let's just normalize length as HIQL does for its targets.
        goal_reps = (
            goal_reps
            / jnp.linalg.norm(goal_reps, axis=-1, keepdims=True)
            * jnp.sqrt(goal_reps.shape[-1])
        )

        # --- Low-Level Action Sampling ---
        low_dist = self.network.select("low_actor")(
            observations, goal_reps, goal_encoded=True, temperature=temperature
        )
        actions = low_dist.sample(seed=low_seed)

        if not self.config["discrete"]:
            actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions. In discrete-action MDPs, this should contain the maximum action value.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = ex_observations
        if config["discrete"]:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        # Define (state-dependent) subgoal representation phi([s; g])
        if config["encoder"] is not None:
            encoder_module = encoder_modules[config["encoder"]]
            goal_rep_seq = [encoder_module()]
        else:
            goal_rep_seq = []
        goal_rep_seq.append(
            MLP(
                hidden_dims=(*config["value_hidden_dims"], config["rep_dim"]),
                activate_final=False,
                layer_norm=config["layer_norm"],
            )
        )
        goal_rep_seq.append(LengthNormalize())
        goal_rep_def = nn.Sequential(goal_rep_seq)

        # Define encoders (Same as HIQL)
        if config["encoder"] is not None:
            value_encoder_def = GCEncoder(
                state_encoder=encoder_module(), concat_encoder=goal_rep_def
            )
            target_value_encoder_def = GCEncoder(
                state_encoder=encoder_module(), concat_encoder=goal_rep_def
            )
            low_actor_encoder_def = GCEncoder(
                state_encoder=encoder_module(), concat_encoder=goal_rep_def
            )
            # High-level actor encoder is handled inside FlowActor or we pass raw features
            # FlowActor takes 'condition'. If pixels, we need to encode 'condition' first.
            # For now, assuming state-based or handling it simply.
            high_actor_encoder_def = None
        else:
            value_encoder_def = GCEncoder(
                state_encoder=Identity(), concat_encoder=goal_rep_def
            )
            target_value_encoder_def = GCEncoder(
                state_encoder=Identity(), concat_encoder=goal_rep_def
            )
            low_actor_encoder_def = GCEncoder(
                state_encoder=Identity(), concat_encoder=goal_rep_def
            )
            high_actor_encoder_def = None

        # Define value and actor networks.
        value_def = GCValue(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            ensemble=True,
            gc_encoder=value_encoder_def,
        )
        target_value_def = GCValue(
            hidden_dims=config["value_hidden_dims"],
            layer_norm=config["layer_norm"],
            ensemble=True,
            gc_encoder=target_value_encoder_def,
        )

        if config["discrete"]:
            low_actor_def = GCDiscreteActor(
                hidden_dims=config["actor_hidden_dims"],
                action_dim=action_dim,
                gc_encoder=low_actor_encoder_def,
            )
        else:
            low_actor_def = GCActor(
                hidden_dims=config["actor_hidden_dims"],
                action_dim=action_dim,
                state_dependent_std=False,
                const_std=config["const_std"],
                gc_encoder=low_actor_encoder_def,
            )

        # --- Flow Actor Definition ---
        high_actor_def = FlowActor(
            hidden_dim=256,  # TODO: Configurable?
            num_blocks=3,
            out_dim=config["rep_dim"],
            dropout_rate=0.1,
            use_layer_norm=config["layer_norm"],
        )

        network_info = dict(
            goal_rep=(
                goal_rep_def,
                (jnp.concatenate([ex_observations, ex_goals], axis=-1)),
            ),
            value=(value_def, (ex_observations, ex_goals)),
            target_value=(target_value_def, (ex_observations, ex_goals)),
            low_actor=(low_actor_def, (ex_observations, ex_goals)),
            high_actor=(
                high_actor_def,
                (
                    jnp.zeros((1, config["rep_dim"])),
                    jnp.zeros((1, 1)),
                    jnp.concatenate([ex_observations, ex_goals], axis=-1),
                ),
            ),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["lr"])
        network_params = network_def.init(init_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params["modules_target_value"] = params["modules_value"]

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name="hiql_fm",  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            expectile=0.7,  # IQL expectile.
            low_alpha=3.0,  # Low-level AWR temperature.
            high_alpha=3.0,  # High-level AWR temperature.
            subgoal_steps=25,  # Subgoal steps.
            rep_dim=10,  # Goal representation dimension.
            low_actor_rep_grad=False,  # Whether low-actor gradients flow to goal representation (use True for pixels).
            const_std=True,  # Whether to use constant standard deviation for the actors.
            discrete=False,  # Whether the action space is discrete.
            encoder=ml_collections.config_dict.placeholder(
                str
            ),  # Visual encoder name (None, 'impala_small', etc.).
            # Dataset hyperparameters.
            dataset_class="HGCDataset",  # Dataset class name.
            value_p_curgoal=0.2,  # Probability of using the current state as the value goal.
            value_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.3,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            gc_negative=True,  # Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as reward.
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(
                int
            ),  # Number of frames to stack.
        )
    )
    return config
