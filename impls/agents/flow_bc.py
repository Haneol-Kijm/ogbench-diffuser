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


def apply_conditioning(x, conditions, action_dim, state_dim=None):
    """
    x: (B, T, C)
    conditions: {t: obs_t, ...} 딕셔너리

    만약 state_dim이 주어지고, conditions의 값이 (state_dim + goal_dim)보다 작다면?
    -> Goal만 업데이트하는 로직이 필요함.

    하지만 더 간단하게:
    conditions 딕셔너리의 값이 '전체(State+Goal)'일 수도 있고, 'Goal만'일 수도 있음.
    이를 구분하기 위해 conditions의 키를 (t, type)으로 하거나,
    값의 shape를 보고 판단?

    가장 확실한 방법:
    conditions = {
        (t, 'full'): val,  # State + Goal 전체 교체
        (t, 'goal'): val   # Goal 부분만 교체
    }
    """
    for key, val in conditions.items():
        if isinstance(key, tuple):
            t, type_ = key
        else:
            t, type_ = key, "full"

        if type_ == "full":
            # 기존 로직: Action 뒤의 모든 것을 교체 (State + Goal)
            x = x.at[:, t, action_dim:].set(val)
        elif type_ == "goal":
            # Goal만 교체: Action + State 뒤의 모든 것을 교체
            # 이를 위해서는 state_dim이 필수
            assert state_dim is not None
            x = x.at[:, t, action_dim + state_dim :].set(val)

    return x


def compute_ot_matching(x0, x1):
    """
    Sinkhorn 알고리즘을 사용한 Minibatch OT 매칭 (JAX/GPU Native).
    x0: (B, D) Noise
    x1: (B, D) Data
    """
    # 1. Cost Matrix (Squared Euclidean)
    # (B, 1, D) - (1, B, D) -> (B, B, D) -> (B, B)
    cost = jnp.sum(jnp.square(x0[:, None, :] - x1[None, :, :]), axis=-1)

    # 2. Sinkhorn Algorithm
    epsilon = 0.1
    n_iters = 50

    def log_sinkhorn_step(carry, _):
        f, g = carry
        g_cost = g[None, :] - cost
        f_new = -epsilon * jax.nn.logsumexp(g_cost / epsilon, axis=1)
        f_cost = f_new[:, None] - cost
        g_new = -epsilon * jax.nn.logsumexp(f_cost / epsilon, axis=0)
        return (f_new, g_new), None

    B = x0.shape[0]
    f = jnp.zeros(B)
    g = jnp.zeros(B)

    (f, g), _ = jax.lax.scan(log_sinkhorn_step, (f, g), None, length=n_iters)

    # 3. Optimal Plan & Matching
    # P_ij = exp((f_i + g_j - C_ij) / eps)
    # 각 데이터(x1)에 대해 가장 확률 높은 노이즈(x0) 인덱스를 찾음
    P = f[:, None] + g[None, :] - cost
    x0_indices = jnp.argmax(P, axis=0)

    x0_sorted = x0[x0_indices]

    return x0_sorted, x1


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

            state_dim = ex_observations.shape[-1]

        # Check if we are doing goal-conditioned BC
        # If so, we need to increase input dim to include goal
        # We assume goal dim == state dim
        # But we don't have goal info in create args easily unless we inspect config or data
        # Let's assume if dataset_class is GCDiffuserSequenceDataset, we have goals.
        # Or simpler: just check config for 'goal_conditioned' flag or similar.
        # For now, let's just add goal_dim if it's in the batch during training?
        # No, model init needs it.

        # Let's check config.
        is_goal_conditioned = config.get("dataset_class") == "GCDiffuserSequenceDataset"

        if is_goal_conditioned:
            # Goal is concatenated to observation
            # We assume goal_dim == state_dim (standard in OGBench)
            input_dim = action_dim + state_dim + state_dim
        else:
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

    def total_loss(self, batch, grad_params, rng=None):
        if rng is None:
            rng = self.rng

        rng, noise_rng, t_rng = jax.random.split(rng, 3)

        # 1. 데이터 준비 (x1)
        # 1. Prepare Data (x1)
        observations = batch["observations"]
        actions = batch["actions"]

        # Handle Goal Conditioning
        if "actor_goals" in batch:
            goals = batch["actor_goals"]
            # Goals are usually (B, G). We need to broadcast to (B, T, G)
            # or if they are already (B, T, G) (unlikely for standard GCDataset logic which gives one goal per episode)
            # GCDiffuserSequenceDataset returns (B, G) for actor_goals.

            if goals.ndim == 2:
                # (B, G) -> (B, 1, G) -> (B, T, G)
                goals = goals[:, None, :]
                goals = jnp.tile(goals, (1, observations.shape[1], 1))

            observations = jnp.concatenate([observations, goals], axis=-1)

        x1 = jnp.concatenate([actions, observations], axis=-1)
        B, T, C = x1.shape

        # 2. Flow Matching Setup (x0)
        x0 = jax.random.normal(noise_rng, shape=x1.shape)

        # ================= [OT 추가됨] =================
        # x0(노이즈)와 x1(데이터) 사이의 거리를 최소화하도록 x0를 재정렬
        # 3차원 (B, T, C)를 (B, T*C)로 펼쳐서 매칭 후 다시 복구
        x0_flat = x0.reshape(B, -1)
        x1_flat = x1.reshape(B, -1)

        x0_flat_matched, _ = compute_ot_matching(x0_flat, x1_flat)

        x0 = x0_flat_matched.reshape(B, T, C)
        # ==============================================

        t = jax.random.uniform(t_rng, shape=(B,), minval=0.0, maxval=1.0)

        # 3. Interpolation (이제 경로는 꼬이지 않은 직선이 됩니다)
        t_expanded = t[:, None, None]
        xt = t_expanded * x1 + (1 - t_expanded) * x0
        ut = x1 - x0

        # 4. Forward Pass
        params = grad_params if grad_params is not None else self.state.params
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

            # state_dim 계산 (TotalDim - ActDim - GoalDim?)
            # GoalDim을 모르지만, conditions에서 유추하거나 인자로 받아야 함.
            # sample 함수에 state_dim 인자 추가 필요.
            # 하지만 sample() 시그니처 바꾸기 번거로우니, 여기서 계산.
            # conditions의 첫 번째 값의 shape를 보면 됨.
            # 하지만 conditions가 이제 (t, type) 키를 가질 수 있음.

            # sample() 호출 시 state_dim을 계산해서 넘겨주는 게 깔끔함.
            # 여기서는 ActDim, ObsDim(State+Goal)을 알고 있음.
            # 하지만 StateDim과 GoalDim을 분리해야 함.
            # ObsDim = StateDim + GoalDim.
            # 보통 StateDim == GoalDim (OGBench).
            # 따라서 StateDim = ObsDim // 2.

            state_dim = ObsDim // 2
            x_next = apply_conditioning(x_next, conditions, ActDim, state_dim)

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
            if goals.ndim == 1:
                goals = goals[None, ...]

            # Broadcast goal to horizon if needed, but here sample_actions takes single obs/goal
            # and sample() handles the loop.
            # But wait, sample() takes 'conditions'.
            # If our model expects (Action + Obs + Goal), then 'conditions' should provide (Obs + Goal).

            # sample() logic:
            # TotalDim = ActDim + ObsDim (where ObsDim includes Goal if conditioned)
            # conditions[0] = observations (which should include goal)

            observations = jnp.concatenate([observations, goals], axis=-1)

        # [Global Goal Conditioning Fix]
        # 1. t=0: State + Goal 모두 고정 (Start Inpainting)
        conditions = {(0, "full"): observations}

        if goals is not None:
            # 2. t=1...H-1: Goal만 고정 (Goal Inpainting)
            # goals: (B, G) -> (B, 1, G) -> Tile to (B, H-1, G) -> Loop set?
            # 딕셔너리에 하나씩 넣어야 함.

            # goals shape: (B, G) (이미 위에서 broadcast 안 된 상태로 가정?
            # 아까 위에서 observations = concat([obs, goals]) 했음.
            # goals 변수는 여전히 (B, G) 또는 (B, 1, G) 상태임.

            # 위에서 goals 차원 처리를 했음:
            # if goals.ndim == 1: goals = goals[None, ...]
            # if goals.ndim == 1: goals = goals[None, ...] (중복)

            # goals는 (B, G) 형태여야 함.
            # 만약 (B, 1, G)라면 squeeze 필요.
            if goals.ndim == 3:
                goals_cond = goals[:, 0, :]  # (B, G)
            else:
                goals_cond = goals  # (B, G)

            for t in range(1, self.horizon):
                conditions[(t, "goal")] = goals_cond

        actions = self.sample(conditions, seed=seed, nfe=32)

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
            dataset_class="GCDiffuserSequenceDataset",  # Default to GC version
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
            # GC Dataset Params (Default)
            value_p_curgoal=0.0,
            value_p_trajgoal=1.0,
            value_p_randomgoal=0.0,
            value_geom_sample=True,
            actor_p_curgoal=0.0,
            actor_p_trajgoal=1.0,
            actor_p_randomgoal=0.0,
            actor_geom_sample=False,
            gc_negative=True,
        )
    )
    return config
