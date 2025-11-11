from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

# --- 헬퍼 함수 포팅 (diffuser/models/helpers.py) ---


def extract(a, t, x_shape):
    """
    diffuser/models/helpers.py (line 150)의 JAX 버전.
    t.shape[0] (배치 크기)을 x_shape[0]으로 사용합니다.
    """
    batch_size = x_shape[0]
    out = a[t]
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    """
    diffuser/models/helpers.py (line 155)의 JAX/Numpy 버전.
    """
    steps = timesteps + 1
    x = jnp.linspace(0, timesteps, steps)
    alphas_cumprod = jnp.cos(((x / timesteps) + s) / (1 + s) * jnp.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return jnp.clip(betas, 0.0001, 0.9999)


def apply_conditioning(x, conditions, action_dim):
    """
    diffuser/models/helpers.py (line 177)의 JAX 버전.
    JAX는 인덱스 할당을 위해 .at[].set()을 사용합니다.
    """
    for t, val in conditions.items():
        # x[:, t, action_dim:] = val.clone()
        x = x.at[:, t, action_dim:].set(val)
    return x


# --- GaussianDiffusion 버퍼/상수 계산 (diffuser/models/diffusion.py __init__) ---


def get_diffuser_buffers(n_timesteps):
    """
    GaussianDiffusion.__init__의 register_buffer 로직을 포팅.
    미리 계산된 상수(버퍼) 딕셔너리를 반환합니다.

    """
    betas = cosine_beta_schedule(n_timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = jnp.cumprod(alphas, axis=0)
    alphas_cumprod_prev = jnp.concatenate([jnp.ones(1), alphas_cumprod[:-1]])

    buffers = {}
    buffers["betas"] = betas
    buffers["alphas_cumprod"] = alphas_cumprod
    buffers["alphas_cumprod_prev"] = alphas_cumprod_prev

    # q(x_t | x_{t-1})
    buffers["sqrt_alphas_cumprod"] = jnp.sqrt(alphas_cumprod)
    buffers["sqrt_one_minus_alphas_cumprod"] = jnp.sqrt(1.0 - alphas_cumprod)
    buffers["log_one_minus_alphas_cumprod"] = jnp.log(1.0 - alphas_cumprod)
    buffers["sqrt_recip_alphas_cumprod"] = jnp.sqrt(1.0 / alphas_cumprod)
    buffers["sqrt_recipm1_alphas_cumprod"] = jnp.sqrt(1.0 / alphas_cumprod - 1)

    # q(x_{t-1} | x_t, x_0)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    buffers["posterior_variance"] = posterior_variance

    # log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
    buffers["posterior_log_variance_clipped"] = jnp.log(
        jnp.clip(posterior_variance, a_min=1e-20)
    )
    buffers["posterior_mean_coef1"] = (
        betas * jnp.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    )
    buffers["posterior_mean_coef2"] = (
        (1.0 - alphas_cumprod_prev) * jnp.sqrt(alphas) / (1.0 - alphas_cumprod)
    )

    # JAX에서는 모든 것을 numpy 배열로 다루므로, to('cuda') 대신
    # Task 4의 에이전트가 이 딕셔너리를 TrainState의 '정적(static)' 멤버로 관리.
    return buffers


# --- Task 3a: 학습 로직 (p_losses) ---


def q_sample(buffers, x_start, t, noise):
    """
    GaussianDiffusion.q_sample의 JAX 버전.
    """
    sample = (
        extract(buffers["sqrt_alphas_cumprod"], t, x_start.shape) * x_start
        + extract(buffers["sqrt_one_minus_alphas_cumprod"], t, x_start.shape) * noise
    )
    return sample


def diffusion_loss_fn(
    unet_apply_fn,  # TemporalUnet.apply
    unet_params,
    buffers,
    loss_weights,  # diffuser.get_loss_weights()로 미리 계산된 가중치
    x_start,  # (B, H, O+A)
    cond,  # {0: (B, O)}
    t,  # (B,)
    rng_key,
    action_dim: int,
    predict_epsilon: bool = False,
):
    """
    GaussianDiffusion.p_losses의 JAX 버전 (Sub-task 3a).

    """
    noise = jax.random.normal(rng_key, x_start.shape)

    x_noisy = q_sample(buffers, x_start, t, noise)
    x_noisy = apply_conditioning(x_noisy, cond, action_dim)

    # 백본 모델(Unet) 호출
    x_recon = unet_apply_fn(unet_params, x_noisy, cond, t)
    x_recon = apply_conditioning(x_recon, cond, action_dim)

    assert noise.shape == x_recon.shape

    # WeightedL2 Loss 계산 (diffuser/models/helpers.py L256)
    if predict_epsilon:
        loss = (x_recon - noise) ** 2
    else:
        loss = (x_recon - x_start) ** 2

    loss = loss * loss_weights
    loss = loss.mean()

    # TODO: a0_loss 등 'info' 딕셔너리 구현 (Task 4에서 필요시 추가)

    return loss, {}  # (loss, info_dict)


# --- Task 3b: 추론 로직 (p_sample_loop) ---


def predict_start_from_noise(buffers, x_t, t, noise):
    """
    GaussianDiffusion.predict_start_from_noise의 JAX 버전.

    """
    return (
        extract(buffers["sqrt_recip_alphas_cumprod"], t, x_t.shape) * x_t
        - extract(buffers["sqrt_recipm1_alphas_cumprod"], t, x_t.shape) * noise
    )


def q_posterior(buffers, x_start, x_t, t):
    """
    GaussianDiffusion.q_posterior의 JAX 버전.

    """
    posterior_mean = (
        extract(buffers["posterior_mean_coef1"], t, x_t.shape) * x_start
        + extract(buffers["posterior_mean_coef2"], t, x_t.shape) * x_t
    )
    posterior_variance = extract(buffers["posterior_variance"], t, x_t.shape)
    posterior_log_variance_clipped = extract(
        buffers["posterior_log_variance_clipped"], t, x_t.shape
    )
    return posterior_mean, posterior_variance, posterior_log_variance_clipped


def p_mean_variance(
    unet_apply_fn,
    unet_params,
    buffers,
    x,
    cond,
    t,
    predict_epsilon: bool = False,
    clip_denoised: bool = True,  # 원본은 False지만 plan_guided.py는 True
):
    """
    GaussianDiffusion.p_mean_variance의 JAX 버전.

    """
    # 백본 모델(Unet) 호출
    pred_noise = unet_apply_fn(unet_params, x, cond, t)

    if predict_epsilon:
        x_recon = predict_start_from_noise(buffers, x, t, pred_noise)
    else:
        x_recon = pred_noise  # 모델이 x0을 직접 예측

    if clip_denoised:
        x_recon = jnp.clip(x_recon, -1.0, 1.0)

    model_mean, posterior_variance, posterior_log_variance = q_posterior(
        buffers, x_start=x_recon, x_t=x, t=t
    )
    return model_mean, posterior_variance, posterior_log_variance


@partial(
    jax.jit,
    static_argnames=(
        "unet_apply_fn",
        "value_apply_fn",
        "action_dim",
        "n_timesteps",
        "predict_epsilon",
        "clip_denoised",
        "guidance_scale",
        "t_stopgrad",
    ),
)
def sample_loop_with_guidance(
    unet_params,
    value_params,
    unet_apply_fn,  # TemporalUnet.apply
    value_apply_fn,  # ValueFunction.apply
    buffers,
    shape,  # (B, H, O+A)
    cond,  # {0: (B, O)}
    rng_key,
    action_dim: int,
    n_timesteps: int,
    guidance_scale: float,
    t_stopgrad: int,
    predict_epsilon: bool = False,
    clip_denoised: bool = True,
):
    """
    GaussianDiffusion.p_sample_loop의 JAX 버전 (Sub-task 3b).
    PyTorch의 'for' 루프를 'jax.lax.scan'으로 변환.
    'plan_guided.py'의 가이던스 로직을 통합.


    """

    # 1. 가이던스 그래디언트 함수 정의
    # 1a. (원본) 값을 반환하는 함수를 먼저 정의
    def value_fn_body(v_params, x_t, cond, t):
        value_pred = value_apply_fn(v_params, x_t, cond, t)
        return value_pred.sum()

    # 1b. [수정된 부분] "함수 공장"을 명시적으로 호출
    #    value_fn_body의 1번 인자(x_t)에 대해 미분하는
    #    새로운 함수 'value_grad_fn'을 생성합니다.
    value_grad_fn = jax.grad(value_fn_body, argnums=1)

    # 2. lax.scan에 사용할 루프 본체(body) 함수
    def loop_body(carry, t_idx):
        # t_idx는 (N-1)부터 0까지 감소
        (x, rng_key) = carry

        # JAX는 배치를 지원하므로 (B,) 크기의 t 벡터 생성
        batch_size = x.shape[0]
        t = jnp.full((batch_size,), t_idx, dtype=jnp.int32)  # make_timesteps

        # 1. 기본 p_mean_variance 계산
        model_mean, model_variance, model_log_variance = p_mean_variance(
            unet_apply_fn,
            unet_params,
            buffers,
            x,
            cond,
            t,
            predict_epsilon,
            clip_denoised,
        )

        # 2. 가이던스 적용 (lax.scan 내부에서 jax.grad 호출)
        # plan_guided.py의 핵심 로직
        grad = value_grad_fn(value_params, x, cond, t)

        stopgrad_mask = (t < t_stopgrad)[:, None, None]
        grad = jnp.where(stopgrad_mask, 0.0, grad)

        model_mean = model_mean + guidance_scale * model_variance * grad

        # 3. 샘플링 (x_{t-1} 계산)
        rng_key, sample_rng = jax.random.split(rng_key)
        noise = jax.random.normal(sample_rng, x.shape)
        # t=0일 때 노이즈 제거
        noise = noise * (t_idx != 0)  # t_idx=0이면 0, 아니면 1

        model_std = jnp.exp(0.5 * model_log_variance)
        x_prev = model_mean + model_std * noise

        # 4. 컨디셔닝 재적용
        x_prev = apply_conditioning(x_prev, cond, action_dim)

        # ( (x_{t-1}, new_rng), None )
        return (x_prev, rng_key), None

    # 3. 루프 시작
    rng_key, init_rng = jax.random.split(rng_key)
    x = jax.random.normal(init_rng, shape)
    x = apply_conditioning(x, cond, action_dim)

    initial_carry = (x, rng_key)
    timesteps = jnp.arange(n_timesteps - 1, -1, -1)  # (N-1) ... 0

    # 4. lax.scan 실행
    (final_x, _), _ = jax.lax.scan(loop_body, initial_carry, timesteps)

    return final_x  # (B, H, O+A)
