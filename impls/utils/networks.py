from typing import Any, Optional, Sequence

import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp


def default_init(scale=1.0):
    """Default kernel initializer."""
    return nn.initializers.variance_scaling(scale, "fan_avg", "uniform")


def ensemblize(cls, num_qs, out_axes=0, **kwargs):
    """Ensemblize a module."""
    return nn.vmap(
        cls,
        variable_axes={"params": 0},
        split_rngs={"params": True},
        in_axes=None,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


class Identity(nn.Module):
    """Identity layer."""

    def __call__(self, x):
        return x


class MLP(nn.Module):
    """Multi-layer perceptron.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        activations: Activation function.
        activate_final: Whether to apply activation to the final layer.
        kernel_init: Kernel initializer.
        layer_norm: Whether to apply layer normalization.
    """

    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
        return x


class LengthNormalize(nn.Module):
    """Length normalization layer.

    It normalizes the input along the last dimension to have a length of sqrt(dim).
    """

    @nn.compact
    def __call__(self, x):
        return x / jnp.linalg.norm(x, axis=-1, keepdims=True) * jnp.sqrt(x.shape[-1])


class Param(nn.Module):
    """Scalar parameter module."""

    init_value: float = 0.0

    @nn.compact
    def __call__(self):
        return self.param("value", init_fn=lambda key: jnp.full((), self.init_value))


class LogParam(nn.Module):
    """Scalar parameter module with log scale."""

    init_value: float = 1.0

    @nn.compact
    def __call__(self):
        log_value = self.param(
            "log_value", init_fn=lambda key: jnp.full((), jnp.log(self.init_value))
        )
        return jnp.exp(log_value)


class TransformedWithMode(distrax.Transformed):
    """Transformed distribution with mode calculation."""

    def mode(self):
        return self.bijector.forward(self.distribution.mode())


class RunningMeanStd(flax.struct.PyTreeNode):
    """Running mean and standard deviation.

    Attributes:
        eps: Epsilon value to avoid division by zero.
        mean: Running mean.
        var: Running variance.
        clip_max: Clip value after normalization.
        count: Number of samples.
    """

    eps: Any = 1e-6
    mean: Any = 1.0
    var: Any = 1.0
    clip_max: Any = 10.0
    count: int = 0

    def normalize(self, batch):
        batch = (batch - self.mean) / jnp.sqrt(self.var + self.eps)
        batch = jnp.clip(batch, -self.clip_max, self.clip_max)
        return batch

    def unnormalize(self, batch):
        return batch * jnp.sqrt(self.var + self.eps) + self.mean

    def update(self, batch):
        batch_mean, batch_var = jnp.mean(batch, axis=0), jnp.var(batch, axis=0)
        batch_count = len(batch)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = m_2 / total_count

        return self.replace(mean=new_mean, var=new_var, count=total_count)


class GCActor(nn.Module):
    """Goal-conditioned actor.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        log_std_min: Minimum value of log standard deviation.
        log_std_max: Maximum value of log standard deviation.
        tanh_squash: Whether to squash the action with tanh.
        state_dependent_std: Whether to use state-dependent standard deviation.
        const_std: Whether to use constant standard deviation.
        final_fc_init_scale: Initial scale of the final fully-connected layer.
        gc_encoder: Optional GCEncoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    log_std_min: Optional[float] = -5
    log_std_max: Optional[float] = 2
    tanh_squash: bool = False
    state_dependent_std: bool = False
    const_std: bool = True
    final_fc_init_scale: float = 1e-2
    gc_encoder: nn.Module = None

    def setup(self):
        self.actor_net = MLP(self.hidden_dims, activate_final=True)
        self.mean_net = nn.Dense(
            self.action_dim, kernel_init=default_init(self.final_fc_init_scale)
        )
        if self.state_dependent_std:
            self.log_std_net = nn.Dense(
                self.action_dim, kernel_init=default_init(self.final_fc_init_scale)
            )
        else:
            if not self.const_std:
                self.log_stds = self.param(
                    "log_stds", nn.initializers.zeros, (self.action_dim,)
                )

    def __call__(
        self,
        observations,
        goals=None,
        goal_encoded=False,
        temperature=1.0,
    ):
        """Return the action distribution.

        Args:
            observations: Observations.
            goals: Goals (optional).
            goal_encoded: Whether the goals are already encoded.
            temperature: Scaling factor for the standard deviation.
        """
        if self.gc_encoder is not None:
            inputs = self.gc_encoder(observations, goals, goal_encoded=goal_encoded)
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
            inputs = jnp.concatenate(inputs, axis=-1)
        outputs = self.actor_net(inputs)

        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        else:
            if self.const_std:
                log_stds = jnp.zeros_like(means)
            else:
                log_stds = self.log_stds

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(
            loc=means, scale_diag=jnp.exp(log_stds) * temperature
        )
        if self.tanh_squash:
            distribution = TransformedWithMode(
                distribution, distrax.Block(distrax.Tanh(), ndims=1)
            )

        return distribution


class GCDiscreteActor(nn.Module):
    """Goal-conditioned actor for discrete actions.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        final_fc_init_scale: Initial scale of the final fully-connected layer.
        gc_encoder: Optional GCEncoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    final_fc_init_scale: float = 1e-2
    gc_encoder: nn.Module = None

    def setup(self):
        self.actor_net = MLP(self.hidden_dims, activate_final=True)
        self.logit_net = nn.Dense(
            self.action_dim, kernel_init=default_init(self.final_fc_init_scale)
        )

    def __call__(
        self,
        observations,
        goals=None,
        goal_encoded=False,
        temperature=1.0,
    ):
        """Return the action distribution.

        Args:
            observations: Observations.
            goals: Goals (optional).
            goal_encoded: Whether the goals are already encoded.
            temperature: Inverse scaling factor for the logits (set to 0 to get the argmax).
        """
        if self.gc_encoder is not None:
            inputs = self.gc_encoder(observations, goals, goal_encoded=goal_encoded)
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
            inputs = jnp.concatenate(inputs, axis=-1)
        outputs = self.actor_net(inputs)

        logits = self.logit_net(outputs)

        distribution = distrax.Categorical(
            logits=logits / jnp.maximum(1e-6, temperature)
        )

        return distribution


class GCValue(nn.Module):
    """Goal-conditioned value/critic function.

    This module can be used for both value V(s, g) and critic Q(s, a, g) functions.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        layer_norm: Whether to apply layer normalization.
        ensemble: Whether to ensemble the value function.
        gc_encoder: Optional GCEncoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    layer_norm: bool = True
    ensemble: bool = True
    gc_encoder: nn.Module = None

    def setup(self):
        mlp_module = MLP
        if self.ensemble:
            mlp_module = ensemblize(mlp_module, 2)
        value_net = mlp_module(
            (*self.hidden_dims, 1), activate_final=False, layer_norm=self.layer_norm
        )

        self.value_net = value_net

    def __call__(self, observations, goals=None, actions=None):
        """Return the value/critic function.

        Args:
            observations: Observations.
            goals: Goals (optional).
            actions: Actions (optional).
        """
        if self.gc_encoder is not None:
            inputs = [self.gc_encoder(observations, goals)]
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
        if actions is not None:
            inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)

        v = self.value_net(inputs).squeeze(-1)

        return v


class GCDiscreteCritic(GCValue):
    """Goal-conditioned critic for discrete actions."""

    action_dim: int = None

    def __call__(self, observations, goals=None, actions=None):
        actions = jnp.eye(self.action_dim)[actions]
        return super().__call__(observations, goals, actions)


class GCBilinearValue(nn.Module):
    """Goal-conditioned bilinear value/critic function.

    This module computes the value function as V(s, g) = phi(s)^T psi(g) / sqrt(d) or the critic function as
    Q(s, a, g) = phi(s, a)^T psi(g) / sqrt(d), where phi and psi output d-dimensional vectors.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        latent_dim: Latent dimension.
        layer_norm: Whether to apply layer normalization.
        ensemble: Whether to ensemble the value function.
        value_exp: Whether to exponentiate the value. Useful for contrastive learning.
        state_encoder: Optional state encoder.
        goal_encoder: Optional goal encoder.
    """

    hidden_dims: Sequence[int]
    latent_dim: int
    layer_norm: bool = True
    ensemble: bool = True
    value_exp: bool = False
    state_encoder: nn.Module = None
    goal_encoder: nn.Module = None

    def setup(self):
        mlp_module = MLP
        if self.ensemble:
            mlp_module = ensemblize(mlp_module, 2)

        self.phi = mlp_module(
            (*self.hidden_dims, self.latent_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
        )
        self.psi = mlp_module(
            (*self.hidden_dims, self.latent_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
        )

    def __call__(self, observations, goals, actions=None, info=False):
        """Return the value/critic function.

        Args:
            observations: Observations.
            goals: Goals.
            actions: Actions (optional).
            info: Whether to additionally return the representations phi and psi.
        """
        if self.state_encoder is not None:
            observations = self.state_encoder(observations)
        if self.goal_encoder is not None:
            goals = self.goal_encoder(goals)

        if actions is None:
            phi_inputs = observations
        else:
            phi_inputs = jnp.concatenate([observations, actions], axis=-1)

        phi = self.phi(phi_inputs)
        psi = self.psi(goals)

        v = (phi * psi / jnp.sqrt(self.latent_dim)).sum(axis=-1)

        if self.value_exp:
            v = jnp.exp(v)

        if info:
            return v, phi, psi
        else:
            return v


class GCDiscreteBilinearCritic(GCBilinearValue):
    """Goal-conditioned bilinear critic for discrete actions."""

    action_dim: int = None

    def __call__(self, observations, goals=None, actions=None, info=False):
        actions = jnp.eye(self.action_dim)[actions]
        return super().__call__(observations, goals, actions, info)


class GCMRNValue(nn.Module):
    """Metric residual network (MRN) value function.

    This module computes the value function as the sum of a symmetric Euclidean distance and an asymmetric
    L^infinity-based quasimetric.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        latent_dim: Latent dimension.
        layer_norm: Whether to apply layer normalization.
        encoder: Optional state/goal encoder.
    """

    hidden_dims: Sequence[int]
    latent_dim: int
    layer_norm: bool = True
    encoder: nn.Module = None

    def setup(self):
        self.phi = MLP(
            (*self.hidden_dims, self.latent_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
        )

    def __call__(self, observations, goals, is_phi=False, info=False):
        """Return the MRN value function.

        Args:
            observations: Observations.
            goals: Goals.
            is_phi: Whether the inputs are already encoded by phi.
            info: Whether to additionally return the representations phi_s and phi_g.
        """
        if is_phi:
            phi_s = observations
            phi_g = goals
        else:
            if self.encoder is not None:
                observations = self.encoder(observations)
                goals = self.encoder(goals)
            phi_s = self.phi(observations)
            phi_g = self.phi(goals)

        sym_s = phi_s[..., : self.latent_dim // 2]
        sym_g = phi_g[..., : self.latent_dim // 2]
        asym_s = phi_s[..., self.latent_dim // 2 :]
        asym_g = phi_g[..., self.latent_dim // 2 :]
        squared_dist = ((sym_s - sym_g) ** 2).sum(axis=-1)
        quasi = jax.nn.relu((asym_s - asym_g).max(axis=-1))
        v = jnp.sqrt(jnp.maximum(squared_dist, 1e-12)) + quasi

        if info:
            return v, phi_s, phi_g
        else:
            return v


class GCIQEValue(nn.Module):
    """Interval quasimetric embedding (IQE) value function.

    This module computes the value function as an IQE-based quasimetric.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        latent_dim: Latent dimension.
        dim_per_component: Dimension of each component in IQE (i.e., number of intervals in each group).
        layer_norm: Whether to apply layer normalization.
        encoder: Optional state/goal encoder.
    """

    hidden_dims: Sequence[int]
    latent_dim: int
    dim_per_component: int
    layer_norm: bool = True
    encoder: nn.Module = None

    def setup(self):
        self.phi = MLP(
            (*self.hidden_dims, self.latent_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
        )
        self.alpha = Param()

    def __call__(self, observations, goals, is_phi=False, info=False):
        """Return the IQE value function.

        Args:
            observations: Observations.
            goals: Goals.
            is_phi: Whether the inputs are already encoded by phi.
            info: Whether to additionally return the representations phi_s and phi_g.
        """
        alpha = jax.nn.sigmoid(self.alpha())
        if is_phi:
            phi_s = observations
            phi_g = goals
        else:
            if self.encoder is not None:
                observations = self.encoder(observations)
                goals = self.encoder(goals)
            phi_s = self.phi(observations)
            phi_g = self.phi(goals)

        x = jnp.reshape(phi_s, (*phi_s.shape[:-1], -1, self.dim_per_component))
        y = jnp.reshape(phi_g, (*phi_g.shape[:-1], -1, self.dim_per_component))
        valid = x < y
        xy = jnp.concatenate(jnp.broadcast_arrays(x, y), axis=-1)
        ixy = xy.argsort(axis=-1)
        sxy = jnp.take_along_axis(xy, ixy, axis=-1)
        neg_inc_copies = jnp.take_along_axis(
            valid, ixy % self.dim_per_component, axis=-1
        ) * jnp.where(ixy < self.dim_per_component, -1, 1)
        neg_inp_copies = jnp.cumsum(neg_inc_copies, axis=-1)
        neg_f = -1.0 * (neg_inp_copies < 0)
        neg_incf = jnp.concatenate(
            [neg_f[..., :1], neg_f[..., 1:] - neg_f[..., :-1]], axis=-1
        )
        components = (sxy * neg_incf).sum(axis=-1)
        v = alpha * components.mean(axis=-1) + (1 - alpha) * components.max(axis=-1)

        if info:
            return v, phi_s, phi_g
        else:
            return v


from typing import Callable

import einops

# --- JAX/Flax로 포팅된 Helper 모듈 ---
# 원본: jannerm/diffuser/diffuser/models/helpers.py

# -----------------------------------------------------------------------------#
# ---------------------------------- modules ----------------------------------#
# -----------------------------------------------------------------------------#


class SinusoidalPosEmb(nn.Module):
    """
    JAX/Flax로 포팅된 SinusoidalPosEmb.
    PyTorch 원본과 로직은 동일합니다.
    """

    dim: int

    @nn.compact
    def __call__(self, x):
        half_dim = self.dim // 2
        div_term = jnp.exp(jnp.arange(half_dim) * -(jnp.log(10000.0) / (half_dim - 1)))
        pe = x[..., None] * div_term
        pe = jnp.concatenate([jnp.sin(pe), jnp.cos(pe)], axis=-1)
        return pe


class Downsample1d(nn.Module):
    """
    JAX/Flax로 포팅된 Downsample1d.
    Conv1d 대신 nn.Conv를 사용합니다.
    """

    dim: int

    @nn.compact
    def __call__(self, x):
        # (B, C, L) -> (B, L, C)
        x_transposed = jnp.transpose(x, (0, 2, 1))

        # nn.Conv(stride=2)
        out = nn.Conv(
            features=self.dim, kernel_size=(3,), strides=(2,), padding="SAME"
        )(x_transposed)

        # (B, L', C) -> (B, C, L')
        return jnp.transpose(out, (0, 2, 1))


class Upsample1d(nn.Module):
    """
    JAX/Flax로 포팅된 Upsample1d. (수정본)
    PyTorch의 nn.ConvTranspose1d에 1:1 대응하는
    Flax의 nn.ConvTranspose를 사용합니다.
    """

    dim: int

    @nn.compact
    def __call__(self, x):
        # (B, C, L) -> (B, L, C)
        # JAX의 nn.ConvTranspose는 (B, L, C) 입력을 기대합니다.
        x_transposed = jnp.transpose(x, (0, 2, 1))

        # PyTorch의 nn.ConvTranspose1d(dim, dim, 4, 2, 1)와 동일한 연산
        # kernel_size=4, stride=2, padding=1
        out = nn.ConvTranspose(
            features=self.dim,
            kernel_size=(4,),
            strides=(2,),
            padding=1,  # PyTorch의 padding=1과 동일하게 작동
        )(x_transposed)

        # (B, L', C) -> (B, C, L')
        # 다시 (B, C, L) 관례로 복귀
        return jnp.transpose(out, (0, 2, 1))


class Conv1dBlock(nn.Module):
    """
    JAX/Flax로 포팅된 Conv1dBlock. (수정본)
    helpers.py의 GroupNorm과 Rearrange 로직을 정확히 반영합니다.
    """

    inp_channels: int
    out_channels: int
    kernel_size: int
    n_groups: int = 8

    @nn.compact
    def __call__(self, x):
        # 원본: nn.Sequential

        # 1. nn.Conv1d(..., padding=kernel_size // 2)
        # (B, C, L) -> (B, L, C)
        x_transposed = jnp.transpose(x, (0, 2, 1))  # (B, L, C_in)

        out = nn.Conv(
            features=self.out_channels,
            kernel_size=(self.kernel_size,),
            padding=self.kernel_size // 2,
        )(
            x_transposed
        )  # (B, L, C_out)

        # 2. GroupNorm 적용을 위해 다시 (B, C, L) 관례로 복귀
        out = jnp.transpose(out, (0, 2, 1))  # (B, C_out, L)

        # 3. Rearrange + GroupNorm + Rearrange
        #    원본은 GroupNorm을 4D 텐서 (B, C, 1, L)에 적용합니다.
        b, c, l = out.shape

        # Rearrange('... -> ... 1 horizon')
        out_reshaped = out.reshape(b, c, 1, l)  # (B, C, 1, L)

        # nn.GroupNorm(n_groups, out_channels)
        # Flax의 nn.GroupNorm은 (..., C) 채널이 마지막에 오는 것을 기대합니다.
        # 따라서 (B, 1, L, C)로 변환
        out_gn_transposed = jnp.transpose(out_reshaped, (0, 2, 3, 1))  # (B, 1, L, C)

        # Flax의 GroupNorm 적용
        out_gn = nn.GroupNorm(num_groups=self.n_groups)(out_gn_transposed)

        # 다시 원본 (B, C, 1, L) 형태로 복원
        out_reshaped = jnp.transpose(out_gn, (0, 3, 1, 2))  # (B, C, 1, L)

        # Rearrange('... 1 horizon -> ... horizon')
        out = out_reshaped.reshape(b, c, l)  # (B, C, L)

        # 4. nn.Mish()
        out = jax.nn.mish(out)

        return out


# -----------------------------------------------------------------------------#
# --------------------------------- attention ---------------------------------#
# -----------------------------------------------------------------------------#


class Residual(nn.Module):
    """
    JAX/Flax로 포팅된 Residual.
    """

    fn: Callable

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class LayerNorm(nn.Module):
    """
    JAX/Flax로 포팅된 diffuser의 커스텀 LayerNorm.
    PyTorch의 nn.Parameter를 Flax의 self.param으로 구현합니다.
    """

    dim: int
    eps: float = 1e-5

    @nn.compact
    def __call__(self, x):
        # g, b 파라미터 생성
        # Flax에서는 (1, C, 1) 대신 (C,)로 저장하고 브로드캐스팅하는 것이 일반적입니다.
        g = self.param("g", nn.initializers.ones, (self.dim,))
        b = self.param("b", nn.initializers.zeros, (self.dim,))

        # (B, C, L) 입력 기준, 채널(C) 축(axis=1)에 대해 var/mean 계산
        # JAX/Flax의 nn.LayerNorm은 마지막 축을 기준으로 하므로,
        # 원본과 동일하게 축 1을 기준으로 직접 구현합니다.
        var = jnp.var(x, axis=1, keepdims=True)
        mean = jnp.mean(x, axis=1, keepdims=True)

        # g와 b를 (1, C, 1)로 브로드캐스팅
        g_broadcasted = g[None, :, None]
        b_broadcasted = b[None, :, None]

        return (x - mean) / jnp.sqrt(var + self.eps) * g_broadcasted + b_broadcasted


class PreNorm(nn.Module):
    """
    JAX/Flax로 포팅된 PreNorm.
    """

    dim: int
    fn: Callable

    norm: Callable = None  # setup에서 정의

    def setup(self):
        self.norm = LayerNorm(self.dim)  # 커스텀 LayerNorm 사용

    @nn.compact
    def __call__(self, x):
        x = self.norm()(x)
        return self.fn(x)


class LinearAttention(nn.Module):
    """
    JAX/Flax로 포팅된 LinearAttention. (수정본)
    원본의 Multi-Head 로직과 einops를 정확히 반영합니다.
    """

    dim: int
    heads: int = 4
    dim_head: int = 32

    scale: float = None
    hidden_dim: int = None
    to_qkv: Callable = None
    to_out: Callable = None

    def setup(self):
        """
        PyTorch의 __init__ 로직을 setup으로 이동합니다.
        """
        self.scale = self.dim_head**-0.5
        self.hidden_dim = self.dim_head * self.heads
        self.to_qkv = nn.Conv(
            features=self.hidden_dim * 3, kernel_size=(1,), use_bias=False
        )
        self.to_out = nn.Conv(features=self.dim, kernel_size=(1,))

    @nn.compact
    def __call__(self, x):
        """
        x는 (B, C, L) 관례를 따름 (C == dim)
        """

        # 1. JAX Conv를 위해 (B, L, C)로 transpose
        x_transposed = jnp.transpose(x, (0, 2, 1))  # (B, L, C)

        # 2. QKV 프로젝션
        # (B, L, C) -> (B, L, H*D*3)
        qkv_transposed = self.to_qkv(x_transposed)

        # 3. PyTorch의 einops (B, C, L) 관례로 복귀
        # (B, L, H*D*3) -> (B, H*D*3, L)
        qkv = jnp.transpose(qkv_transposed, (0, 2, 1))

        # 4. Q, K, V로 분리 (채널 축 기준)
        # 원본: .chunk(3, dim=1)
        qkv_list = jnp.array_split(qkv, 3, axis=1)  # 각각 (B, H*D, L)

        # 5. Head 차원 분리
        # 리스트 컴프리헨션 (map(lambda...)와 동일)
        q, k, v = [
            einops.rearrange(t, "b (h c) l -> b h c l", h=self.heads) for t in qkv_list
        ]

        # 6. 원본 로직 (Scale, Softmax, Einsum)
        q = q * self.scale

        k = nn.softmax(k, axis=-1)  # L (length) 축에 대해 소프트맥스
        context = jnp.einsum(
            "b h d n, b h e n -> b h d e", k, v
        )  # (B, H, D_head, D_head)

        out = jnp.einsum("b h d e, b h d n -> b h e n", context, q)  # (B, H, D_head, L)
        out = einops.rearrange(out, "b h c l -> b (h c) l")  # (B, H*D, L)

        # 8. JAX Conv를 위해 (B, L, C)로 transpose
        out_transposed = jnp.transpose(out, (0, 2, 1))  # (B, L, H*D)

        # 9. 최종 프로젝션
        out_proj = self.to_out(out_transposed)  # (B, L, C_final)

        # 10. (B, C, L) 관례로 복귀
        return jnp.transpose(out_proj, (0, 2, 1))


# --- JAX/Flax로 포팅된 메인 모델 ---
# 원본: jannerm/diffuser/diffuser/models/temporal.py


class ResidualTemporalBlock(nn.Module):
    """
    JAX/Flax로 포팅된 ResidualTemporalBlock.
    PyTorch의 nn.ModuleList는 Flax의 setup()에서 Python 리스트로 구현합니다.
    """

    inp_channels: int
    out_channels: int
    embed_dim: int
    horizon: int
    kernel_size: int = 5

    blocks: Sequence[Callable] = None
    time_mlp: Callable = None
    residual_conv: Callable = None

    def setup(self):
        # PyTorch의 nn.ModuleList
        self.blocks = [
            Conv1dBlock(self.inp_channels, self.out_channels, self.kernel_size),
            Conv1dBlock(self.out_channels, self.out_channels, self.kernel_size),
        ]

        # PyTorch의 nn.Sequential
        self.time_mlp = nn.Sequential(
            [
                nn.mish,
                nn.Linear(self.embed_dim),
                # Rearrange('batch t -> batch t 1') 대신 expand_dims
            ]
        )

        # PyTorch의 nn.Conv1d(1) 또는 nn.Identity
        self.residual_conv = (
            nn.Conv(features=self.out_channels, kernel_size=(1,))
            if self.inp_channels != self.out_channels
            else lambda x: x
        )

    def __call__(self, x, t):
        """
        x : [ B x C x H ] (batch, transition_dim, horizon)
        t : [ B x D ] (batch, embed_dim)
        """
        out = self.blocks[0](x)  # (B, C_out, H)

        # time_mlp 처리
        t_emb = self.time_mlp(t)  # (B, C_out)
        t_emb = t_emb[:, :, None]  # (B, C_out) -> (B, C_out, 1)

        out = out + t_emb  # (B, C_out, H) + (B, C_out, 1) -> 브로드캐스팅

        out = self.blocks[1](out)  # (B, C_out, H)

        # residual_conv 처리
        # (B, C, H) -> (B, H, C)
        x_transposed = jnp.transpose(x, (0, 2, 1))

        if self.inp_channels != self.out_channels:
            res = self.residual_conv(x_transposed)  # (B, H, C_out)
            res = jnp.transpose(res, (0, 2, 1))  # (B, C_out, H)
        else:
            res = self.residual_conv(x)  # Identity

        return out + res


class TemporalUnet(nn.Module):
    """
    JAX/Flax로 포팅된 TemporalUnet.
    PyTorch의 __init__ 로직이 setup()으로 이동했습니다.
    """

    horizon: int
    transition_dim: int
    cond_dim: int
    dim: int = 32
    dim_mults: Sequence[int] = (1, 2, 4, 8)
    attention: bool = False

    # Flax에서 모듈 리스트는 setup에서 동적으로 생성
    time_mlp: Callable = None
    downs: Sequence = None
    ups: Sequence = None
    mid_block1: Callable = None
    mid_attn: Callable = None
    mid_block2: Callable = None
    final_conv: Callable = None

    def setup(self):
        dims = [self.transition_dim, *map(lambda m: self.dim * m, self.dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = self.dim
        self.time_mlp = nn.Sequential(
            [
                SinusoidalPosEmb(self.dim),
                nn.Linear(self.dim * 4),
                nn.mish,
                nn.Linear(self.dim),
            ]
        )

        self.downs = []
        self.ups = []
        num_resolutions = len(in_out)

        current_horizon = self.horizon

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                [
                    ResidualTemporalBlock(
                        dim_in, dim_out, embed_dim=time_dim, horizon=current_horizon
                    ),
                    ResidualTemporalBlock(
                        dim_out, dim_out, embed_dim=time_dim, horizon=current_horizon
                    ),
                    (
                        Residual(PreNorm(LinearAttention()))
                        if self.attention
                        else lambda x: x
                    ),
                    Downsample1d(dim_out) if not is_last else lambda x: x,
                ]
            )

            if not is_last:
                current_horizon = current_horizon // 2

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(
            mid_dim, mid_dim, embed_dim=time_dim, horizon=current_horizon
        )
        self.mid_attn = (
            Residual(PreNorm(LinearAttention())) if self.attention else lambda x: x
        )
        self.mid_block2 = ResidualTemporalBlock(
            mid_dim, mid_dim, embed_dim=time_dim, horizon=current_horizon
        )

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                [
                    ResidualTemporalBlock(
                        dim_out * 2, dim_in, embed_dim=time_dim, horizon=current_horizon
                    ),
                    ResidualTemporalBlock(
                        dim_in, dim_in, embed_dim=time_dim, horizon=current_horizon
                    ),
                    (
                        Residual(PreNorm(LinearAttention()))
                        if self.attention
                        else lambda x: x
                    ),
                    Upsample1d(dim_in) if not is_last else lambda x: x,
                ]
            )

            if not is_last:
                current_horizon = current_horizon * 2

        self.final_conv = nn.Sequential(
            [
                Conv1dBlock(self.dim, self.dim, kernel_size=5),
                # nn.Conv1d(dim, transition_dim, 1)
                nn.Conv(features=self.transition_dim, kernel_size=(1,)),
            ]
        )

    def __call__(self, x, cond, time):
        """
        x : [ B x H x T ] (batch, horizon, transition_dim)
        cond: [ B x C_cond ] (batch, cond_dim) - 포팅 시 cond 처리는 생략 (원본도 사용 안 함)
        time: [ B ] (batch,)
        """

        # PyTorch: x = einops.rearrange(x, 'b h t -> b t h')
        # (B, H, T) -> (B, T, H)
        x = jnp.transpose(x, (0, 2, 1))

        t = self.time_mlp(time)
        h = []

        # downs 리스트는 setup에서 생성된 Python 리스트
        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # ups 리스트는 setup에서 생성된 Python 리스트
        for resnet, resnet2, attn, upsample in self.ups:
            # PyTorch: x = torch.cat((x, h.pop()), dim=1)
            x = jnp.concatenate((x, h.pop()), axis=1)  # (B, C, L) 기준 C (channel) 축
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            x = upsample(x)

        # final_conv는 nn.Conv를 포함하므로 (B, L, C) 입력을 받아야 함
        # (B, C, L) -> (B, L, C)
        x_transposed = jnp.transpose(x, (0, 2, 1))
        x_out = self.final_conv(x_transposed)  # (B, L, C_out)
        # (B, L, C_out) -> (B, C_out, L)
        x = jnp.transpose(x_out, (0, 2, 1))

        # PyTorch: x = einops.rearrange(x, 'b t h -> b h t')
        # (B, T, H) -> (B, H, T)
        x = jnp.transpose(x, (0, 2, 1))

        return x
