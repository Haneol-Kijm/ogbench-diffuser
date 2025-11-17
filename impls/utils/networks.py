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


from typing import Callable, List

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
            padding="SAME",  # PyTorch의 padding=1과 동일하게 작동
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

    def setup(self):
        # PyTorch의 nn.ModuleList
        self.blocks = [
            Conv1dBlock(self.inp_channels, self.out_channels, self.kernel_size),
            Conv1dBlock(self.out_channels, self.out_channels, self.kernel_size),
        ]

        # PyTorch의 nn.Sequential
        self.time_mlp = nn.Sequential(
            [
                jax.nn.mish,
                nn.Dense(self.out_channels),
                lambda x: x[:, :, None],
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

        out = self.blocks[0](x) + self.time_mlp(
            t
        )  # (B, C_out, H) + (B, C_out, 1) -> 브로드캐스팅
        out = self.blocks[1](out)  # (B, C_out, H)

        if self.inp_channels != self.out_channels:
            # residual_conv 처리
            # (B, C, H) -> (B, H, C)
            x_transposed = jnp.transpose(x, (0, 2, 1))
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

    def setup(self):
        dims = [self.transition_dim, *map(lambda m: self.dim * m, self.dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f"[ models/temporal ] Channel dimensions: {in_out}")

        time_dim = self.dim
        self.time_mlp = nn.Sequential(
            [
                SinusoidalPosEmb(self.dim),
                nn.Dense(self.dim * 4),
                jax.nn.mish,
                nn.Dense(self.dim),
            ]
        )

        downs = []
        ups = []
        num_resolutions = len(in_out)
        print(in_out)

        current_horizon = self.horizon

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            downs.append(
                [
                    ResidualTemporalBlock(
                        dim_in, dim_out, embed_dim=time_dim, horizon=current_horizon
                    ),
                    ResidualTemporalBlock(
                        dim_out, dim_out, embed_dim=time_dim, horizon=current_horizon
                    ),
                    (
                        Residual(PreNorm(dim_out, LinearAttention(dim_out)))
                        if self.attention
                        else lambda x: x
                    ),
                    Downsample1d(dim_out) if not is_last else lambda x: x,
                ]
            )

            if not is_last:
                current_horizon = current_horizon // 2

        self.downs = tuple(downs)

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(
            mid_dim, mid_dim, embed_dim=time_dim, horizon=current_horizon
        )
        self.mid_attn = (
            Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
            if self.attention
            else lambda x: x
        )
        self.mid_block2 = ResidualTemporalBlock(
            mid_dim, mid_dim, embed_dim=time_dim, horizon=current_horizon
        )

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            ups.append(
                [
                    ResidualTemporalBlock(
                        dim_out * 2, dim_in, embed_dim=time_dim, horizon=current_horizon
                    ),
                    ResidualTemporalBlock(
                        dim_in, dim_in, embed_dim=time_dim, horizon=current_horizon
                    ),
                    (
                        Residual(PreNorm(dim_in, LinearAttention(dim_in)))
                        if self.attention
                        else lambda x: x
                    ),
                    Upsample1d(dim_in) if not is_last else lambda x: x,
                ]
            )

            if not is_last:
                current_horizon = current_horizon * 2

        self.ups = tuple(ups)

        self.final_conv = nn.Sequential(
            [
                Conv1dBlock(self.dim, self.dim, kernel_size=5),
                lambda x: jnp.transpose(x, (0, 2, 1)),
                # nn.Conv1d(dim, transition_dim, 1)
                nn.Conv(features=self.transition_dim, kernel_size=(1,)),
                lambda x: jnp.transpose(x, (0, 2, 1)),
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
            # skip-connection of U-Net
            x = jnp.concatenate((x, h.pop()), axis=1)  # (B, C, L) 기준 C (channel) 축
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            x = upsample(x)

        # final_conv는 nn.Conv를 포함하므로 (B, L, C) 입력을 받아야 함
        x = self.final_conv(x)

        # PyTorch: x = einops.rearrange(x, 'b t h -> b h t')
        # (B, T, H) -> (B, H, T)
        x = jnp.transpose(x, (0, 2, 1))

        return x


class ValueFunction(nn.Module):
    horizon: int
    transition_dim: int
    cond_dim: int
    dim: int = 32
    dim_mults: Sequence[int] = (1, 2, 4, 8)
    out_dim: int = 1

    def setup(self):
        horizon_curr = self.horizon

        dims = [self.transition_dim, *map(lambda m: self.dim * m, self.dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.time_mlp = nn.Sequential(
            [
                SinusoidalPosEmb(self.dim),
                nn.Dense(self.dim * 4),
                jax.nn.mish,
                nn.Dense(self.dim),
            ]
        )

        blocks = []
        num_resolutions = len(in_out)

        print(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            block_modules = (
                ResidualTemporalBlock(
                    dim_in,
                    dim_out,
                    kernel_size=5,
                    embed_dim=self.dim,
                    horizon=horizon_curr,
                ),
                ResidualTemporalBlock(
                    dim_out,
                    dim_out,
                    kernel_size=5,
                    embed_dim=self.dim,
                    horizon=horizon_curr,
                ),
                Downsample1d(dim_out),
            )
            blocks.append(block_modules)

            if not is_last:
                horizon_curr = horizon_curr // 2

        self.blocks = tuple(blocks)

        mid_dim = dims[-1]
        mid_dim_2 = mid_dim // 2
        mid_dim_3 = mid_dim // 4

        self.mid_block1 = ResidualTemporalBlock(
            mid_dim,
            mid_dim_2,
            kernel_size=5,
            embed_dim=self.dim,
            horizon=horizon_curr,
        )
        self.mid_down1 = Downsample1d(mid_dim_2)
        horizon_curr = horizon_curr // 2

        self.mid_block2 = ResidualTemporalBlock(
            mid_dim_2,
            mid_dim_3,
            kernel_size=5,
            embed_dim=self.dim,
            horizon=horizon_curr,
        )
        self.mid_down2 = Downsample1d(mid_dim_3)
        horizon_curr = horizon_curr // 2
        ##
        fc_dim = mid_dim_3 * max(horizon_curr, 1)

        # PyTorch의 nn.Sequential을 Flax의 개별 레이어로 정의
        # 입력 차원은 flatten된 x(fc_dim)와 t(self.time_dim)가 합쳐진 것
        self.final_layer1 = nn.Dense(fc_dim // 2)
        self.final_layer2 = nn.Dense(self.out_dim)

    def __call__(self, x, cond, time, *args):
        # (PyTorch forward 로직 시작)

        # (B, L, C) -> (B, C, L)
        # PyTorch의 nn.Conv1d 로직을 1:1 포팅하기 위해 축을 변경합니다.
        x = einops.rearrange(x, "b h t -> b t h")

        # Time MLP 실행
        t = self.time_mlp(time)

        # Downsampling Blocks 실행
        for resnet, resnet2, downsample in self.blocks:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = downsample(x)

        # Mid Blocks 실행
        x = self.mid_block1(x, t)
        x = self.mid_down1(x)
        ##
        x = self.mid_block2(x, t)
        x = self.mid_down2(x)
        # Flatten
        x = x.reshape((x.shape[0], -1))

        # (PyTorch forward의 torch.cat 부분)
        x_t = jnp.concatenate([x, t], axis=-1)

        # Final MLP 실행
        # (PyTorch forward의 self.final_block(torch.cat(...)) 부분)
        x = self.final_layer1(x_t)
        x = jax.nn.mish(x)
        out = self.final_layer2(x)

        return out


# ==============================================================================
# "flow_guidance" (PyTorch) DiT 아키텍처 1:1 JAX/Flax 이식 코드
# 원본: ai4science-westlakeu/flow_guidance/.../models_flow/transformer.py
# 대상: seohongpark/ogbench/impls/utils/networks.py
# ==============================================================================

from functools import partial
from typing import Tuple


#################################################################################
#                    Transformer Blocks from DiT                                #
#################################################################################
class Mlp(nn.Module):
    """DiT의 MLP 블록 (GELU 사용)"""

    in_features: int
    hidden_features: Optional[int] = None
    out_features: Optional[int] = None
    act_layer: Any = nn.gelu
    bias: bool = True
    drop: float = 0.0
    use_conv: bool = False  # (ogbench 스타일에서는 사용되지 않음)

    @nn.compact
    def __call__(self, x, deterministic: bool):
        out_features = self.out_features or self.in_features
        hidden_features = self.hidden_features or self.in_features

        # JAX의 nn.Dense는 bias=True가 기본값
        x = nn.Dense(hidden_features, kernel_init=default_init())(x)
        x = self.act_layer(x)
        x = nn.Dropout(rate=self.drop)(x, deterministic=deterministic)
        x = nn.Dense(out_features, kernel_init=default_init())(x)
        x = nn.Dropout(rate=self.drop)(x, deterministic=deterministic)
        return x


class Attention(nn.Module):
    """DiT의 Multi-Head Attention (qkv 수동 구현)"""

    dim: int
    num_heads: int = 8
    qkv_bias: bool = False
    attn_drop: float = 0.0
    proj_drop: float = 0.0

    @nn.compact
    def __call__(self, x, deterministic: bool):
        B, N, C = x.shape
        assert self.dim % self.num_heads == 0, "dim should be divisible by num_heads"
        head_dim = self.dim // self.num_heads
        scale = head_dim**-0.5

        # qkv를 3*dim으로 한 번에 계산
        qkv = nn.Dense(
            self.dim * 3, use_bias=self.qkv_bias, kernel_init=default_init()
        )(x)
        # (B, N, 3 * C) -> (B, N, 3, num_heads, head_dim) -> (3, B, num_heads, N, head_dim)
        qkv = qkv.reshape(B, N, 3, self.num_heads, head_dim).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, num_heads, N, head_dim)

        # Scaled Dot-Product Attention
        attn = (q @ k.transpose(0, 1, 3, 2)) * scale  # (B, num_heads, N, N)
        attn = nn.softmax(attn, axis=-1)
        attn = nn.Dropout(rate=self.attn_drop)(attn, deterministic=deterministic)

        # (B, num_heads, N, N) @ (B, num_heads, N, head_dim) -> (B, num_heads, N, head_dim)
        x = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, C)

        x = nn.Dense(self.dim, kernel_init=default_init())(x)
        x = nn.Dropout(rate=self.proj_drop)(x, deterministic=deterministic)
        return x


def modulate(x, shift, scale):
    """adaLN-Zero의 핵심 연산. (B, N, C) * (B, 1, C) + (B, 1, C)"""
    return x * (1 + scale[:, None, :]) + shift[:, None, :]


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
def get_1d_sincos_pos_embed_from_grid_jax(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = jnp.arange(embed_dim // 2, dtype=jnp.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = jnp.einsum("m,d->md", pos, omega)  # (M, D/2)

    emb_sin = jnp.sin(out)
    emb_cos = jnp.cos(out)

    emb = jnp.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed_jax(embed_dim, grid_size, dtype=jnp.float32):
    grid = jnp.arange(grid_size, dtype=dtype)
    pos_embed = get_1d_sincos_pos_embed_from_grid_jax(embed_dim, grid)  # (T, D)
    return pos_embed


#################################################################################
#                   Core Transformer Layers from DiT                            #
#################################################################################
class Layer(nn.Module):
    """DiT의 핵심 트랜스포머 블록 (adaLN-Zero 적용)"""

    hidden_size: int
    num_heads: int
    mlp_ratio: float = 4.0
    attn_drop: float = 0.0
    proj_drop: float = 0.0

    @nn.compact
    def __call__(self, x, c, deterministic: bool):
        # x: (B, N, C_hidden), c: (B, C_hidden) (시간 임베딩)

        # adaLN-Zero: t 임베딩 'c'로 6개의 파라미터(shift/scale/gate * 2) 예측
        # (B, C_hidden) -> (B, 6 * C_hidden)
        mod_params = nn.Sequential(
            [
                nn.silu,
                nn.Dense(
                    6 * self.hidden_size, kernel_init=default_init(0.0)
                ),  # 원본은 0으로 초기화
            ]
        )(c)

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = jnp.split(
            mod_params, 6, axis=1
        )

        # 1. Self-Attention (with adaLN)
        x_norm1 = nn.LayerNorm(use_scale=False, use_bias=False)(x)
        x_mod1 = modulate(x_norm1, shift_msa, scale_msa)

        attn_block = Attention(
            dim=self.hidden_size,
            num_heads=self.num_heads,
            attn_drop=self.attn_drop,
            proj_drop=self.proj_drop,
        )
        attn_out = attn_block(x_mod1, deterministic=deterministic)

        # gate_msa를 (B, 1, 1)이 아닌 (B, 1, C)로 브로드캐스팅
        x = x + gate_msa[:, None, :] * attn_out

        # 2. MLP (with adaLN)
        x_norm2 = nn.LayerNorm(use_scale=False, use_bias=False)(x)
        x_mod2 = modulate(x_norm2, shift_mlp, scale_mlp)

        mlp_block = Mlp(
            in_features=self.hidden_size,
            hidden_features=int(self.hidden_size * self.mlp_ratio),
            act_layer=nn.gelu,
            drop=0.0,
        )
        mlp_out = mlp_block(x_mod2, deterministic=deterministic)

        x = x + gate_mlp[:, None, :] * mlp_out

        return x


class FinalLayer(nn.Module):
    """DiT의 최종 출력 레이어"""

    hidden_size: int
    out_channels: int

    @nn.compact
    def __call__(self, x, c):
        # c: (B, C_hidden)
        mod_params = nn.Sequential(
            [
                nn.silu,
                nn.Dense(
                    2 * self.hidden_size, kernel_init=default_init(0.0)
                ),  # 원본은 0으로 초기화
            ]
        )(c)

        shift, scale = jnp.split(mod_params, 2, axis=1)

        x_norm = nn.LayerNorm(use_scale=False, use_bias=False)(x)
        x_mod = modulate(x_norm, shift, scale)
        x_out = nn.Dense(self.out_channels, kernel_init=default_init(0.0))(
            x_mod
        )  # 원본은 0으로 초기화

        return x_out


class DiT_TimestepEmbedder(nn.Module):
    """DiT 스타일 Sinusoidal 시간 임베딩 MLP"""

    hidden_size: int
    frequency_embedding_size: int = 256
    max_period: int = 10000

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        # t: (B,)
        half = dim // 2
        freqs = jnp.exp(
            -jnp.log(max_period)
            * jnp.arange(start=0, stop=half, dtype=jnp.float32)
            / half
        )
        # t: (B,) -> (B, 1), freqs: (half,) -> (1, half)
        args = t[:, None] * freqs[None, :]
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1)
        if dim % 2:
            embedding = jnp.concatenate(
                [embedding, jnp.zeros_like(embedding[:, :1])], axis=-1
            )
        return embedding

    @nn.compact
    def __call__(self, t):
        t_freq = self.timestep_embedding(
            t, self.frequency_embedding_size, self.max_period
        )
        t_emb = nn.Sequential(
            [
                nn.Dense(self.hidden_size, kernel_init=default_init()),
                nn.silu,
                nn.Dense(self.hidden_size, kernel_init=default_init()),
            ]
        )(t_freq)
        return t_emb


# seohongpark/ogbench/impls/utils/networks.py


class TimeSeriesEmbedder(nn.Module):
    """1D 시계열 임베더 (Conv1d 또는 Linear)"""

    seq_len: int = 224
    in_channels: int = 3
    embed_dim: int = 768
    # [수정 2] flatten과 norm_layer 로직 부활
    flatten: bool = True
    # norm_layer는 Flax에서 보통 모듈로 전달하지 않고, 필요하면 내부에서 정의하거나
    # 플래그(use_norm)로 처리합니다. 원본이 None(Identity)만 쓰므로 여기선 생략하되
    # flatten 로직은 아래 __call__에 반영했습니다.
    bias: bool = True
    proj: str = "conv"
    conv_k: int = 1

    @nn.compact
    def __call__(self, x):
        # x: (B, C, T) - 원본 PyTorch 입력 순서 가정
        B, C, T = x.shape
        assert (
            T == self.seq_len
        ), f"Input seq len ({T}) doesn't match model ({self.seq_len})."

        # Proj 구현
        if self.proj == "conv":
            # Conv1d: (B, C, T) -> (B, T, C)로 바꿔서 JAX Conv 적용
            x_t = x.transpose(0, 2, 1)
            # padding='SAME'은 PyTorch의 padding='same'과 동일
            x_embed = nn.Conv(
                features=self.embed_dim,
                kernel_size=(self.conv_k,),
                padding="SAME",
                use_bias=self.bias,
                kernel_init=nn.initializers.xavier_uniform(),
            )(
                x_t
            )  # (B, T, D)

            # 원본의 Conv1d는 (B, D, T)를 뱉지만, JAX Conv는 (B, T, D)를 뱉음.
            # 원본 로직 일치를 위해 (B, D, T)로 다시 돌려놓음 (개념상)
            x_embed = x_embed.transpose(0, 2, 1)  # (B, D, T)

        elif self.proj == "linear":
            # Linear: (B, C, T) -> (B, T, C) -> Dense -> (B, T, D) -> (B, D, T)
            x_t = x.transpose(0, 2, 1)
            x_embed = nn.Dense(
                features=self.embed_dim,
                use_bias=self.bias,
                kernel_init=nn.initializers.xavier_uniform(),
            )(
                x_t
            )  # (B, T, D)
            x_embed = x_embed.transpose(0, 2, 1)  # (B, D, T)

        # Flatten 로직 (1:1 재현)
        if self.flatten:
            # (B, D, T) -> (B, T, D)
            x_embed = x_embed.transpose(0, 2, 1)

        # Norm Layer는 원본에서 None이므로 생략 (Identity)

        return x_embed


class DiT_Transformer(nn.Module):
    """DiT 아키텍처 본체"""

    seq_len: int
    in_channels: int
    out_channels: Optional[int] = None
    hidden_size: int = 1152
    depth: int = 28
    num_heads: int = 16
    mlp_ratio: float = 4.0
    x_emb_proj: str = "conv"
    x_emb_proj_conv_k: int = 1

    @nn.compact
    def __call__(self, x, t, deterministic: bool):
        # x: (B, C_in, T), t: (B,)

        # 1. 시간 임베딩
        t_emb = DiT_TimestepEmbedder(hidden_size=self.hidden_size)(t)  # (B, D)
        c = t_emb  # 컨디션

        # 2. 입력 임베딩 + 위치 임베딩
        x_embedder = TimeSeriesEmbedder(
            seq_len=self.seq_len,
            in_channels=self.in_channels,
            embed_dim=self.hidden_size,
            proj=self.x_emb_proj,
            conv_k=self.x_emb_proj_conv_k,
        )
        x_embed = x_embedder(x)  # (B, T, D)

        # 위치 임베딩 초기화 (원본 initialize_weights 로직)
        pos_embed_data = get_1d_sincos_pos_embed_jax(self.hidden_size, self.seq_len)
        pos_embed = self.param(
            "pos_embed",
            lambda key, shape, dtype: pos_embed_data[None, ...],  # (1, T, D)
            (1, self.seq_len, self.hidden_size),
            jnp.float32,
        )

        x = x_embed + pos_embed  # (B, T, D)

        # 3. 트랜스포머 블록 (DiT Layer)
        for _ in range(self.depth):
            x = Layer(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
            )(
                x, c, deterministic=deterministic
            )  # (B, T, D)

        # 4. 최종 레이어
        final_out_channels = self.out_channels or self.in_channels
        x_out = FinalLayer(
            hidden_size=self.hidden_size, out_channels=final_out_channels
        )(
            x, c
        )  # (B, T, C_out)

        # 원본은 (B, C, T)로 permute해서 반환
        return x_out.transpose(0, 2, 1)  # (B, C_out, T)


#################################################################################
#                Wrapper for Flow Matching and Diffuser                         #
#################################################################################
class TransformerFlow(nn.Module):
    """DiT를 Flow Matching 백본으로 사용하기 위한 래퍼"""

    seq_len: int
    in_channels: int
    out_channels: Optional[int] = None
    hidden_size: int = 1152
    depth: int = 28
    num_heads: int = 16
    mlp_ratio: float = 4.0
    x_emb_proj: str = "conv"
    x_emb_proj_conv_k: int = 1

    @nn.compact
    def __call__(self, x, t, deterministic: bool = True):
        # x: (B, T, C_in) - JAX/ogbench 표준 입력
        # t: (B,)

        b, t_len, c_in = x.shape

        # JAX (B, T, C) -> PyTorch (B, C, T)
        x_t = x.transpose(0, 2, 1)

        # t가 (B, 1)이나 (B, T) 등으로 들어올 경우 (B,)로 축소
        # (JAX에서는 보통 (B,)로 잘 들어옴)
        if t.ndim > 1:
            t = t[..., 0]
        if t.ndim == 0:
            t = jnp.repeat(t, b)

        # DiT 모델 호출 (입/출력 모두 (B, C, T) 형식)
        dit_model = DiT_Transformer(
            seq_len=self.seq_len,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            hidden_size=self.hidden_size,
            depth=self.depth,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            x_emb_proj=self.x_emb_proj,
            x_emb_proj_conv_k=self.x_emb_proj_conv_k,
        )
        x_out_t = dit_model(x_t, t, deterministic=deterministic)  # (B, C_out, T)

        # PyTorch (B, C, T) -> JAX (B, T, C)
        x_out = x_out_t.transpose(0, 2, 1)

        return x_out
