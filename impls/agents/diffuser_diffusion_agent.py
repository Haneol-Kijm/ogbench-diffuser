from typing import Any, Dict

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax

# Task 3에서 포팅한 유틸리티
from utils.diffuser_utils import (  # Task 3b의 핵심 추론 함수
    apply_conditioning,
    get_diffuser_buffers,
    q_sample,
    sample_loop_with_guidance,
)

# ogbench 유틸리티
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field, restore_agent

# Task 1에서 포팅한 모델
from utils.networks import TemporalUnet, ValueFunction


class DiffuserDiffusionAgent(flax.struct.PyTreeNode):
    """
    [Task 4b] diffuser의 TemporalUnet(확산 모델)을 학습/추론하는 에이전트.
    ogbench/agents/gcbc.py 스타일을 따름
    """

    rng: Any
    network: TrainState  # TemporalUnet의 TrainState (학습 대상)
    value_network: Any = nonpytree_field()  # ValueFunction의 apply_fn과 동결된 파라미터

    config: Dict = nonpytree_field()
    buffers: Dict = nonpytree_field()
    loss_weights: Any = nonpytree_field()  # (B, H, O+A)

    # --- [수정] Loss 함수를 클래스 메소드로 정의 ---
    def diffusion_loss(self, batch, grad_params, rng_key):
        """
        Task 3a의 diffuser_utils.diffusion_loss_fn을 메소드로 구현.
        diffuser/models/diffusion.py의 p_losses 로직
        """
        rng, t_rng, noise_rng = jax.random.split(rng_key, 3)

        # 1. 타임스텝 't' 샘플링
        batch_size = batch["trajectories"].shape[0]
        t = jax.random.randint(
            t_rng, (batch_size,), 0, self.config["n_diffusion_steps"], dtype=jnp.int32
        )

        x_start = batch["trajectories"]

        # 2. 노이즈 주입 (q_sample)
        noise = jax.random.normal(noise_rng, x_start.shape)
        x_noisy = q_sample(self.buffers, x_start, t, noise)
        x_noisy = apply_conditioning(
            x_noisy, batch["conditions"], self.config["action_dim"]
        )

        # 3. Unet 예측
        x_recon = self.network.select("unet")(
            x_noisy,
            batch["conditions"],
            t,
            params=grad_params,
        )
        x_recon = apply_conditioning(
            x_recon, batch["conditions"], self.config["action_dim"]
        )

        # 4. 'l2' 손실 계산
        #
        if self.config["predict_epsilon"]:
            loss_target = noise
        else:
            loss_target = x_start

        loss = (x_recon - loss_target) ** 2

        # 5. 가중치 적용 (WeightedLoss)
        #
        loss = (loss * self.loss_weights).mean()

        # TODO: a0_loss 등 info 딕셔너리 구현
        info = {"loss": loss}

        return loss, info

    # --- 1. Loss (ogbench 스타일) ---
    @jax.jit
    def total_loss(self, batch: Dict[str, Any], grad_params: Any, rng: Any = None):
        """
        ogbench/main.py의 검증(validation) 루프용

        """
        params = grad_params if grad_params is not None else self.network.params
        rng = rng if rng is not None else self.rng

        rng, loss_rng = jax.random.split(rng)

        loss, info = self.diffusion_loss(batch, grad_params, loss_rng)
        return loss, info

    @jax.jit
    def update(self, batch: Dict[str, Any]):
        """
        gcbc.py의 apply_loss_fn 방식을 따름
        """
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, goals, seed, temperature=0.0):
        """가이드 샘플링(추론) 실행. ogbench/utils/evaluation.py가 호출함"""
        rng = jax.random.PRNGKey(seed)

        # 1. 입력 변환 (ogbench -> diffuser)
        batch_size = observations.shape[0]
        # Diffuser 모델은 (B, O)가 아닌 (B, H, O+A) 입력을 가정하고 컨디셔닝함
        # 평가 시점의 컨디션: 시작(t=0)과 끝(t=H-1)
        cond = {0: observations, self.config["horizon"] - 1: goals}  # (B, O)  # (B, O)

        shape = (batch_size, self.config["horizon"], self.config["transition_dim"])

        # 2. [Task 3b] 가이드 샘플링 루프 실행
        trajectories = sample_loop_with_guidance(
            unet_params=self.network.params,  # Unet 파라미터
            value_params=self.value_network.params,  # 동결된 Value 파라미터
            unet_apply_fn=self.network.apply_fn,  # Unet apply (ModuleDict)
            value_apply_fn=self.value_network.apply_fn,  # Value apply (ModuleDict)
            buffers=self.buffers,
            shape=shape,
            cond=cond,
            rng_key=rng,
            action_dim=self.config["action_dim"],
            n_timesteps=self.config["n_diffusion_steps"],
            guidance_scale=self.config["guidance_scale"],
            t_stopgrad=self.config["t_stopgrad"],
            predict_epsilon=self.config["predict_epsilon"],
            clip_denoised=self.config["clip_denoised"],
        )

        # 3. 결과 변환 (diffuser -> ogbench)
        # Receding Horizon Planning: 전체 궤적(plan)에서 첫 액션만 반환
        #
        actions = trajectories[:, 0, : self.config["action_dim"]]

        return actions

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
        rng, unet_rng, value_rng = jax.random.split(rng, 3)

        # 1. config에 데이터 의존적 파라미터 추가
        # (DiffuserSequenceDataset은 (B, H, A+O)를 반환)
        transition_dim = ex_observations.shape[-1]
        action_dim = ex_actions.shape[-1]
        cond_dim = transition_dim - action_dim

        config["transition_dim"] = transition_dim
        config["action_dim"] = action_dim
        config["cond_dim"] = cond_dim

        # 2. 모델 정의 (Task 1)
        unet_def = TemporalUnet(
            horizon=config["horizon"],
            transition_dim=transition_dim,
            cond_dim=cond_dim,
            dim=config["dim"],
            dim_mults=config["dim_mults"],
            attention=config["attention"],
        )

        value_def = ValueFunction(
            horizon=config["horizon"],
            transition_dim=transition_dim,
            cond_dim=cond_dim,
            dim=config["dim_value"],  # 밸류 전용 dim 사용 (config에 추가 필요)
            dim_mults=config["dim_mults_value"],
            out_dim=1,
        )

        # 3. Unet(학습 대상) TrainState 생성 (GCBC 스타일)
        ex_x = ex_observations
        ex_cond = {0: ex_observations[:, 0, action_dim:]}  # (B, O)
        ex_t = jnp.ones((ex_observations.shape[0],), dtype=jnp.int32)

        network_info = dict(
            unet=(unet_def, (ex_x, ex_cond, ex_t)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config["learning_rate"])
        network_params = network_def.init(unet_rng, **network_args)["params"]
        network = TrainState.create(network_def, network_params, tx=network_tx)

        # 4. ValueFunction(동결 대상) 파라미터 로드
        if config.get("value_checkpoint_path") is None:
            raise ValueError("Config must provide 'value_checkpoint_path'")

        # 4a. ValueFunction의 TrainState 뼈대 생성 (복원용)
        value_network_info = dict(
            value=(value_def, (ex_x, ex_cond, ex_t)),
        )
        value_networks = {
            k: v[0] for k, v in value_network_info.items()
        }  # {'value': value_def}
        value_init_args = {
            k: v[1] for k, v in value_network_info.items()
        }  # {'value': (...)}

        value_network_def = ModuleDict(value_networks)
        value_network_params = value_network_def.init(value_rng, **value_init_args)[
            "params"
        ]
        dummy_value_state = TrainState.create(
            value_network_def,
            value_network_params,
            tx=network_tx,  # tx는 아무거나 상관 없음
        )

        # 4b. 가중치 복원
        print(f"Loading ValueFunction from: {config['value_checkpoint_path']}")
        restored_value_state = restore_agent(
            dummy_value_state,
            config["value_checkpoint_path"],
            config["value_checkpoint_epoch"],
        )
        # 동결된 파라미터와 apply_fn을 value_network로 묶음
        value_network = restored_value_state.replace(tx=None)  # 옵티마이저 제거

        # 5. 확산 버퍼 및 손실 가중치 로드
        buffers = get_diffuser_buffers(config["n_diffusion_steps"])

        # diffuser.get_loss_weights 로직
        #
        loss_weights = jnp.ones((config["horizon"], transition_dim), dtype=jnp.float32)
        discounts = config["loss_discount"] ** jnp.arange(
            config["horizon"], dtype=jnp.float32
        )
        discounts = discounts / discounts.mean()
        # 브로드캐스팅을 위해 (H,) -> (H, 1)
        loss_weights = loss_weights * discounts[:, None]
        loss_weights = loss_weights.at[0, :action_dim].set(config["action_weight"])

        return cls(
            rng=rng,
            network=network,
            value_network=value_network,  # 동결된 TrainState(params+apply_fn)
            config=flax.core.FrozenDict(**config),
            buffers=buffers,
            loss_weights=loss_weights,
        )


# --- 5. Get Config (ogbench 스타일) ---
def get_config():
    """
    ogbench/main.py가 로드할 기본 설정.
    diffuser/config/locomotion.py의 'diffusion' 딕셔너리 기준

    """
    config = ml_collections.ConfigDict(
        dict(
            # Agent
            agent_name="diffuser_diffusion",
            learning_rate=2e-4,
            # Dataset (DiffuserSequenceDataset)
            dataset_class="DiffuserSequenceDataset",
            normalizer="GaussianNormalizer",  # locomotion.py 기본값
            max_path_length=1000,
            use_padding=True,
            frame_stack=None,
            p_aug=0.0,
            discrete=False,
            encoder=None,
            # Model (TemporalUnet)
            horizon=32,
            dim=32,
            dim_mults=(1, 2, 4, 8),
            attention=False,
            # Model (ValueFunction) - 평가용
            # locomotion.py는 동일한 dim을 썼지만, 분리 가능하도록 별도 지정
            dim_value=32,
            dim_mults_value=(1, 2, 4, 8),
            value_checkpoint_path=ml_collections.config_dict.placeholder(str),
            value_checkpoint_epoch=ml_collections.config_dict.placeholder(int),
            # Diffusion (GaussianDiffusion)
            n_diffusion_steps=20,
            action_weight=10,
            loss_discount=1.0,
            predict_epsilon=False,  # locomotion.py 기본값
            clip_denoised=True,  # 추론 시 필수
            # Guidance (plan_guided.py / sampling/functions.py)
            #
            guidance_scale=0.1,  # 논문 기본값
            t_stopgrad=0,
            # training.py 하이퍼파라미터 (main.py가 오버라이드)
            batch_size=32,
            n_train_steps=1e6,  # 원본 스텝 수
        )
    )
    return config
