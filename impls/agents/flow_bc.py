from functools import partial
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import ml_collections
import optax
from flax import struct
from utils.flax_utils import TrainState
from utils.networks import TransformerFlow

# ==============================================================================
# 1. Helper Function
# ==============================================================================


def apply_conditioning(x, conditions, action_dim):
    """
    x: (B, T, C)
    conditions: {t: obs_t, ...} 딕셔너리
    """
    for t, val in conditions.items():
        x = x.at[:, t, action_dim:].set(val)
    return x


# ==============================================================================
# 2. Flow BC Agent (ogbench Standard Style)
# ==============================================================================


class FlowBCAgent(struct.PyTreeNode):
    state: TrainState
    rng: Any

    # Static fields
    horizon: int = struct.field(pytree_node=False)
    action_dim: int = struct.field(pytree_node=False)

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        if ex_actions.ndim == 3:
            action_dim = ex_actions.shape[-1]
            state_dim = ex_observations.shape[-1]
        else:
            action_dim = ex_actions.shape[-1]
            state_dim = ex_observations.shape[-1]

        input_dim = action_dim + state_dim
        horizon = config["horizon"]

        network_def = TransformerFlow(
            seq_len=horizon,
            in_channels=input_dim,
            out_channels=input_dim,
            hidden_size=config["hidden_size"],
            depth=config["depth"],
            num_heads=config["num_heads"],
            mlp_ratio=config["mlp_ratio"],
            x_emb_proj=config["x_emb_proj"],
            x_emb_proj_conv_k=config["x_emb_proj_conv_k"],
        )

        # Init params
        mock_input = jnp.zeros((1, horizon, input_dim))
        mock_t = jnp.zeros((1,))
        variables = network_def.init(init_rng, mock_input, mock_t, deterministic=True)
        params = variables["params"]

        tx = optax.adam(learning_rate=config["learning_rate"])

        state = TrainState.create(model_def=network_def, params=params, tx=tx)

        return cls(state=state, rng=rng, horizon=horizon, action_dim=action_dim)

    # [추가] ogbench 표준 인터페이스: Loss 계산 로직 분리
    def total_loss(self, batch, grad_params, rng=None):
        # main.py에서 호출할 때는 rng가 없을 수 있음 -> self.rng 사용
        if rng is None:
            rng = self.rng

        # RNG 분할 (Noise, Time)
        rng, noise_rng, t_rng = jax.random.split(rng, 3)

        # 1. 데이터 준비
        observations = batch["observations"]
        actions = batch["actions"]
        x1 = jnp.concatenate([actions, observations], axis=-1)  # (B, T, C)
        B, T, C = x1.shape

        # 2. Flow Matching Setup
        x0 = jax.random.normal(noise_rng, shape=x1.shape)
        t = jax.random.uniform(t_rng, shape=(B,), minval=0.0, maxval=1.0)

        # 3. Interpolation
        t_expanded = t[:, None, None]
        xt = t_expanded * x1 + (1 - t_expanded) * x0
        ut = x1 - x0  # Target Velocity

        # 4. Forward Pass
        # grad_params가 있으면 그것을 사용 (Gradient 계산용), 없으면 self.state.params 사용 (Validation용)
        params = grad_params if grad_params is not None else self.state.params

        # self.state(...) 호출 시 params 인자를 전달
        vt = self.state(xt, t, deterministic=False, params=params)

        # 5. MSE Loss
        loss = jnp.mean(jnp.square(vt - ut))

        return loss, {"loss": loss}

    @jax.jit
    def update(self, batch):
        new_rng, step_rng = jax.random.split(self.rng)

        # loss_fn 래퍼: apply_loss_fn이 요구하는 시그니처 (params -> loss, info)
        def loss_fn(params):
            # total_loss에 현재 rng와 params를 전달
            return self.total_loss(batch, grad_params=params, rng=step_rng)

        # TrainState의 apply_loss_fn을 통해 Gradient 계산 및 업데이트 수행
        new_state, info = self.state.apply_loss_fn(loss_fn)

        return self.replace(state=new_state, rng=new_rng), info

    @partial(jax.jit, static_argnames=("nfe",))
    def sample(
        self,
        conditions: Dict[int, jnp.ndarray],
        seed: Optional[jax.Array] = None,
        nfe: int = 32,
    ):
        first_cond = list(conditions.values())[0]
        B, _ = first_cond.shape

        Horizon = self.horizon
        ActDim = self.action_dim
        ObsDim = first_cond.shape[-1]
        TotalDim = ActDim + ObsDim

        if seed is not None:
            rng = seed
        else:
            rng = self.rng

        rng, sample_rng = jax.random.split(rng)
        x = jax.random.normal(sample_rng, shape=(B, Horizon, TotalDim))
        dt = 1.0 / nfe

        def step_fn(carry, t_idx):
            x = carry
            t_val = t_idx.astype(jnp.float32) / nfe
            t_batch = jnp.full((B,), t_val)

            v_pred = self.state(x, t_batch, deterministic=True)
            x_next = x + v_pred * dt
            x_next = apply_conditioning(x_next, conditions, ActDim)

            return x_next, None

        x_final, _ = jax.lax.scan(step_fn, x, jnp.arange(nfe))
        actions = x_final[:, 0, :ActDim]

        return actions

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        if observations.ndim == 1:
            observations = observations[None, ...]
            squeeze_output = True
        else:
            squeeze_output = False

        if goals is not None:
            if goals.ndim == 1:
                goals = goals[None, ...]
            observations = jnp.concatenate([observations, goals], axis=-1)

        conditions = {0: observations}

        actions = self.sample(conditions, seed=seed, nfe=32)

        if squeeze_output:
            actions = actions[0]

        return actions


# ==============================================================================
# 3. Get Config
# ==============================================================================


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name="flow_bc",
            learning_rate=3e-4,
            dataset_class="DiffuserSequenceDataset",
            normalizer="GaussianNormalizer",
            max_path_length=1000,
            use_padding=True,
            frame_stack=None,
            p_aug=0.0,
            discrete=False,
            encoder=None,
            # Model Hyperparameters (RL Small)
            horizon=32,
            hidden_size=256,
            depth=6,
            num_heads=4,
            mlp_ratio=4.0,
            x_emb_proj="conv",
            x_emb_proj_conv_k=1,
            n_train_steps=1e6,
            batch_size=256,
            nfe=32,
            clip_denoised=False,
            guidance_scale=1.0,
        )
    )
    return config
