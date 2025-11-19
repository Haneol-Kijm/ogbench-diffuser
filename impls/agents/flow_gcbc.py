from typing import Any, Optional, Tuple

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax
from flax import linen as nn
from flax.training import train_state

from utils.flax_utils import nonpytree_field
from utils.networks import TransformerFlow

@flax.struct.dataclass
class FlowGCBCAgent(flax.struct.PyTreeNode):
    """Flow Matching Goal-Conditioned Behavioral Cloning (FlowGCBC) agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: Any,
        action_space: Any,
        config: Any,
    ):
        """Create a new FlowGCBCAgent."""
        """Create a new FlowGCBCAgent."""
        # Input: Action + State + Goal
        # We assume Goal dim == State dim
        action_dim = action_space.shape[-1]
        state_dim = observation_space.shape[-1]
        
        # Update config with action_dim
        config = config.unlock()
        config['action_dim'] = action_dim
        config = flax.core.FrozenDict(config)

        
        in_channels = action_dim + state_dim + state_dim
        out_channels = action_dim

        # TransformerFlow expects (B, T, C) input
        # We use T=1 for policy
        network_def = TransformerFlow(
            seq_len=1,
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_size=config['hidden_size'],
            depth=config['depth'],
            num_heads=config['num_heads'],
        )

        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        # Dummy input for initialization: (B, T, C)
        dummy_input = jnp.zeros((1, 1, in_channels))
        dummy_t = jnp.zeros((1,))
        
        params = network_def.init(init_rng, dummy_input, dummy_t)['params']

        network = train_state.TrainState.create(
            apply_fn=network_def.apply,
            params=params,
            tx=optax.adam(config['learning_rate']),
        )

        return cls(rng=rng, network=network, config=config)

    def total_loss(self, batch, grad_params, rng=None):
        """Compute the Flow Matching loss."""
        # 1. Prepare Data
        # x1: Target Action (B, A) -> (B, 1, A)
        x1 = batch['actions'][:, None, :]
        
        # Condition: State + Goal (B, S+G) -> (B, 1, S+G)
        observations = batch['observations'][:, None, :]
        goals = batch['actor_goals'][:, None, :]
        cond = jnp.concatenate([observations, goals], axis=-1)
        
        batch_size = x1.shape[0]
        
        # 2. Sample Noise and Time
        rng = rng if rng is not None else self.rng
        rng, step_rng = jax.random.split(rng)
        
        # x0: Noise (B, 1, A)
        x0 = jax.random.normal(step_rng, x1.shape)
        
        # t: Time (B,)
        t = jax.random.uniform(step_rng, (batch_size,))
        
        # 3. Interpolation (Conditional Flow Matching)
        # xt = t * x1 + (1 - t) * x0
        t_reshaped = t[:, None, None]
        xt = t_reshaped * x1 + (1 - t_reshaped) * x0
        
        # 4. Model Forward
        # Input: Concatenate xt and cond -> (B, 1, A+S+G)
        model_input = jnp.concatenate([xt, cond], axis=-1)
        
        # vt: Vector field prediction (B, 1, A)
        params = grad_params if grad_params is not None else self.network.params
        vt = self.network.apply_fn({'params': params}, model_input, t)
        
        # 5. Target Vector Field
        # ut = x1 - x0
        ut = x1 - x0
        
        # 6. Loss (MSE)
        loss = jnp.mean((vt - ut) ** 2)
        
        return loss, {'loss': loss}

    @jax.jit
    def update(self, batch):
        """Update the agent."""
        new_rng, rng = jax.random.split(self.rng)
        
        def loss_fn(params):
            return self.total_loss(batch, params, rng=rng)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, info), grads = grad_fn(self.network.params)
        new_network = self.network.apply_gradients(grads=grads)

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        """Sample actions using ODE solver (Euler method)."""
        # observations: (B, S) or (S,)
        # goals: (B, G) or (G,)
        
        if observations.ndim == 1:
            observations = observations[None, ...]
            squeeze_output = True
        else:
            squeeze_output = False
            
        if goals is not None:
            if goals.ndim == 1:
                goals = goals[None, ...]
        
        batch_size = observations.shape[0]
        
        # Prepare Condition: (B, 1, S+G)
        obs_expanded = observations[:, None, :]
        goals_expanded = goals[:, None, :]
        cond = jnp.concatenate([obs_expanded, goals_expanded], axis=-1)
        
        # Initialize x0 (Noise): (B, 1, A)
        if seed is not None:
            if isinstance(seed, int):
                rng = jax.random.PRNGKey(seed)
            else:
                rng = seed
        else:
            rng = self.rng
        x = jax.random.normal(rng, (batch_size, 1, self.config['action_dim'])) * temperature
        
        # Euler Integration
        # We use a fixed number of steps (e.g., 10) for inference
        num_steps = self.config['nfe']
        dt = 1.0 / num_steps
        
        def body_fn(i, x):
            t = jnp.full((batch_size,), i * dt)
            
            # Model Input: (B, 1, A+S+G)
            model_input = jnp.concatenate([x, cond], axis=-1)
            
            # Predict Vector Field
            v = self.network.apply_fn({'params': self.network.params}, model_input, t)
            
            # Update x
            return x + v * dt

        x = jax.lax.fori_loop(0, num_steps, body_fn, x)
        
        # Return Action: (B, A)
        # Clip to action space [-1, 1] if necessary, but usually handled by env wrapper
        # Here we just return the raw output
        x = x.squeeze(1)
        
        if squeeze_output:
            x = x[0]
            
        return x


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='flow_gcbc',  # Agent name.
            learning_rate=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            hidden_size=256,  # Hidden size.
            depth=4,  # Depth.
            num_heads=4,  # Number of heads.
            mlp_ratio=4.0,  # MLP ratio.
            x_emb_proj='conv',  # Embedding projection type.
            x_emb_proj_conv_k=1,  # Embedding projection kernel size.
            clip_denoised=False,  # Whether to clip denoised values.
            guidance_scale=1.0,  # Guidance scale.
            nfe=10,  # Number of function evaluations for sampling.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name.

            # Dataset hyperparameters.
            dataset_class='GCDataset',  # Dataset class name.
            discount=0.99,  # Discount factor.
            discrete=False,  # Whether the action space is discrete.
            value_p_curgoal=0.0,  # Probability of using the current state as the value goal.
            value_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.0,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            gc_negative=True,  # Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as reward.
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
        )
    )
    return config
