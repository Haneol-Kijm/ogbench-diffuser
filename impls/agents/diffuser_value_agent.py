from typing import Any, Dict

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax

# Task 3에서 포팅한 유틸리티
from utils.diffuser_utils import apply_conditioning, get_diffuser_buffers, q_sample

# ogbench 유틸리티
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field

# Task 1에서 포팅한 모델
from utils.networks import ValueFunction

# --- 내부 손실 함수 (Task 3 로직을 에이전트 내부로 이동) ---
# diffuser/models/helpers.py의 ValueLoss 로직


class DiffuserValueAgent(flax.struct.PyTreeNode):
    """
    [Task 4a] diffuser의 ValueFunction을 학습시키는 에이전트.
    ogbench/agents/gcbc.py 스타일을 따름
    """

    rng: Any
    network: TrainState  # ValueFunction의 TrainState
    config: Dict = nonpytree_field()

    # 확산 스케줄 버퍼
    buffers: Dict = nonpytree_field()

    def value_loss(self, batch, grad_params, rng=None):
        """
        DiffuserValueAgent의 내부 손실 함수. (메소드로 변경)
        diffuser/models/diffusion.py의 ValueDiffusion.p_losses 로직
        """
        rng = rng if rng is not None else self.rng
        rng, t_rng, noise_rng = jax.random.split(rng, 3)

        # 1. 타임스텝 't' 샘플링
        batch_size = batch["trajectories"].shape[0]
        t = jax.random.randint(
            t_rng,
            (batch_size,),
            0,
            self.config["n_diffusion_steps"],  # self.config에서 직접 참조
            dtype=jnp.int32,
        )

        # 2. 노이즈 주입 (q_sample)
        noise = jax.random.normal(noise_rng, batch["trajectories"].shape)
        x_noisy = q_sample(
            self.buffers, batch["trajectories"], t, noise
        )  # self.buffers에서 참조
        x_noisy = apply_conditioning(
            x_noisy,
            batch["conditions"],
            self.config["action_dim"],  # self.config에서 참조
        )

        # 3. ValueFunction 예측
        pred_values = self.network.apply_fn(  # self.state에서 참조
            grad_params, x_noisy, batch["conditions"], t
        ).squeeze(
            -1
        )  # (B, 1) -> (B,)

        target_values = batch["values"]  # (B,)

        # 4. 'value_l2' 손실 계산
        loss = ((pred_values - target_values) ** 2).mean()

        # 5. 메트릭(info) 계산
        safe_pred = jnp.nan_to_num(pred_values, 0.0)
        safe_targ = jnp.nan_to_num(target_values, 0.0)
        corr = jnp.corrcoef(safe_pred, safe_targ)[0, 1]
        corr = jnp.nan_to_num(corr, 0.0)

        info = {
            "loss": loss,
            "mean_pred": pred_values.mean(),
            "mean_targ": target_values.mean(),
            "min_pred": pred_values.min(),
            "max_pred": pred_values.max(),
            "min_targ": target_values.min(),
            "max_targ": target_values.max(),
            "correlation": corr,
        }
        return loss, info

    # --- 1. Loss (ogbench 스타일) ---
    @jax.jit
    def total_loss(self, batch: Dict[str, Any], grad_params, rng=None):
        """
        ogbench/main.py의 검증(validation) 루프용
        """
        rng = rng if rng is not None else self.rng

        rng, value_rng = jax.random.split(rng)
        loss, info = self.value_loss(batch, grad_params, rng=value_rng)
        return loss, info

    # --- 2. Update (ogbench 스타일) ---
    @jax.jit
    def update(self, batch: Dict[str, Any]):

        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

        return self.replace(state=new_network, rng=new_rng), info

    # --- 3. Sample Actions (ogbench 스타일) ---
    @jax.jit
    def sample_actions(self, observations, goals, seed):
        """
        이 에이전트는 추론(planning) 기능이 없습니다.
        ogbench/main.py의 평가 루프가 실패하지 않도록
        더미(dummy) 액션을 반환합니다.
        """
        batch_size = observations.shape[0]
        dummy_actions = jnp.zeros((batch_size, self.config["action_dim"]))
        return dummy_actions

    # --- 4. Create (ogbench 스타일) ---
    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions. In discrete-action MDPs, this should contain the maximum action value.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        # 1. config에 데이터 의존적 파라미터 추가
        # ex_observations는 (B, H, O)가 아니라 (B, H, O+A) (DiffuserDataset 기준)
        transition_dim = ex_observations.shape[-1]
        # ex_actions는 (B, H, A)
        action_dim = ex_actions.shape[-1]
        # cond_dim은 diffuser 원본에서 obs_dim
        cond_dim = transition_dim - action_dim

        config["transition_dim"] = transition_dim
        config["action_dim"] = action_dim
        config["cond_dim"] = cond_dim

        # 2. 모델(ValueFunction) 초기화 (Task 1)
        model_def = ValueFunction(
            horizon=config["horizon"],
            transition_dim=transition_dim,
            cond_dim=cond_dim,
            dim=config["dim"],
            dim_mults=config["dim_mults"],
            out_dim=1,
        )

        # 2a. 더미 입력 정의
        dummy_x = ex_observations
        dummy_cond = {0: ex_observations[:, 0, action_dim:]}  # (B, O)
        dummy_t = jnp.ones((ex_observations.shape[0],), dtype=jnp.int32)

        # 2b. GCBC의 network_info 딕셔너리 생성
        # (ValueFunction.apply는 (x, cond, t) 순서로 인자를 받음)
        network_info = dict(
            value=(model_def, (dummy_x, dummy_cond, dummy_t)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        # 3. 옵티마이저 및 TrainState 생성
        network_def = ModuleDict(networks)  # {'value': model_def}
        network_tx = optax.adam(learning_rate=config["lr"])

        # 3a. ModuleDict가 init을 통해 파라미터 자동 생성
        # (network_args는 {'value': (dummy_x, dummy_cond, dummy_t)})
        network_params = network_def.init(init_rng, **network_args)["params"]

        # 3b. TrainState 생성
        network = TrainState.create(network_def, network_params, tx=network_tx)

        # --- ▲ 수정 완료 ▲ ---

        # 4. 확산 버퍼 로드 (q_sample에 필요)
        buffers = get_diffuser_buffers(config["n_diffusion_steps"])

        return cls(
            rng=rng,
            network=network,
            config=flax.core.FrozenDict(**config),
            buffers=buffers,
        )


# --- 5. Get Config (ogbench 스타일) ---
def get_config():
    """
    ogbench/main.py가 로드할 기본 설정.
    diffuser/config/locomotion.py의 'values' 딕셔너리 기준

    """
    config = ml_collections.ConfigDict(
        dict(
            # Agent
            agent_name="diffuser_value",
            lr=2e-4,
            # Dataset (DiffuserValueDataset)
            dataset_class="DiffuserValueDataset",
            normalizer="GaussianNormalizer",  # locomotion.py 기본값
            max_path_length=1000,
            # Model (ValueFunction)
            horizon=32,
            dim=32,
            dim_mults=(1, 2, 4, 8),
            # Diffusion (ValueDiffusion)
            n_diffusion_steps=20,
            discount=0.99,  # (ValueDataset이 사용)
            # training.py 하이퍼파라미터 (main.py가 오버라이드)
            batch_size=32,
            n_train_steps=200e3,  # 원본 스텝 수
        )
    )
    return config
