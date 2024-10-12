from typing import List

import jax
import jax.numpy as jnp
from jax.ops import segment_sum
import flax.linen as nn
from flax.linen.initializers import glorot_uniform, zeros_init


def compute_fuzzy_laplacian(edge_index, theta, num_nodes, edge_weight=None, add_self_loop=False):
    assert edge_index[0].shape[0] == theta.shape[0] 
    if edge_weight is not None:
        assert edge_index[0].shape[0] == edge_weight.shape[0]
        
    senders, receivers = edge_index
    edge_weight = edge_weight if edge_weight is not None else jnp.ones_like(edge_index[0])

    edge_director_src_to_tgt = jnp.exp(1j*theta)
    edge_director_tgt_to_src = jnp.exp(1j*(jnp.pi/2 - theta))

    conv_senders = jnp.concatenate((senders, receivers))
    conv_receivers = jnp.concatenate((receivers, senders))
    edge_director = jnp.concatenate((edge_director_src_to_tgt, edge_director_tgt_to_src))
    edge_weight = jnp.tile(edge_weight, 2)

    if add_self_loop:
        conv_senders = jnp.concatenate((conv_senders, jnp.arange(num_nodes)))
        conv_receivers = jnp.concatenate((conv_receivers, jnp.arange(num_nodes)))
        edge_weight = jnp.concatenate((edge_weight, jnp.ones(num_nodes)))
        edge_director = jnp.concatenate((edge_director, jnp.array([1+1j]*num_nodes)))

    out_weight = jnp.real(edge_director)**2 * edge_weight
    in_weight = jnp.imag(edge_director)**2 * edge_weight
    
    deg_senders = jax.ops.segment_sum(out_weight, conv_senders, num_nodes) + 1e-12
    deg_receivers = jax.ops.segment_sum(in_weight, conv_senders, num_nodes) + 1e-12

    deg_inv_sqrt_senders = jnp.where(deg_senders>1e11, 0.0, jax.lax.rsqrt(deg_senders))
    deg_inv_sqrt_receivers = jnp.where(deg_receivers>1e11, 0.0, jax.lax.rsqrt(deg_receivers))

    edge_weight_src_to_tgt = deg_inv_sqrt_senders[conv_senders] * \
                             out_weight * \
                             deg_inv_sqrt_receivers[conv_receivers]

    edge_weight_tgt_to_src = deg_inv_sqrt_receivers[conv_senders] * \
                             in_weight * \
                             deg_inv_sqrt_senders[conv_receivers]

    return (conv_senders, conv_receivers), \
           (edge_weight_src_to_tgt.reshape(-1, 1), edge_weight_tgt_to_src.reshape(-1, 1))


class FuzzyDirGCNConv(nn.Module):
    out_channels: int
    improved: bool = False
    use_bias: bool = True
    self_feature_transform: bool = False

    def setup(self):
        self.aggr = lambda x, index, num_nodes: segment_sum(
            data=x, segment_ids=index, num_segments=num_nodes,
            indices_are_sorted=False, unique_indices=False)

        self.lin_src_to_dst = nn.Dense(
            self.out_channels, kernel_init=glorot_uniform(), use_bias=False)

        self.lin_dst_to_src = nn.Dense(
            self.out_channels, kernel_init=glorot_uniform(), use_bias=False)

        if self.self_feature_transform:
            self.lin_self = nn.Dense(
                self.out_channels, kernel_init=glorot_uniform(), use_bias=False)

        if self.use_bias:
            self.bias_src_to_dst = self.param(
                "bias_src_to_dst", zeros_init(), (self.out_channels,), jnp.float32)
            self.bias_dst_to_src = self.param(
                "bias_dst_to_src", zeros_init(), (self.out_channels,), jnp.float32)
            if self.self_feature_transform:
                self.bias_self = self.param(
                    "bias_self", zeros_init(), (self.out_channels,), jnp.float32)

    def __call__(self,
        x: jnp.ndarray,
        edge_index: jnp.ndarray,
        edge_weight: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        edge_weight should have been computed from an enclosing model
        via `compute_fuzzy_laplacian` function
        """

        num_nodes = x.shape[0]
        senders, receivers = edge_index
        edge_weight_src_to_tgt, edge_weight_tgt_to_src = edge_weight

        x_src_to_dst = self.aggr(
            x[senders] * edge_weight_src_to_tgt,
            receivers,
            num_nodes)

        x_dst_to_src = self.aggr(
            x[senders] * edge_weight_tgt_to_src,
            receivers,
            num_nodes)

        x_src_to_dst = self.lin_src_to_dst(x_src_to_dst)
        x_dst_to_src = self.lin_dst_to_src(x_dst_to_src)

        if self.use_bias:
            x_src_to_dst = x_src_to_dst + self.bias_src_to_dst
            x_dst_to_src = x_dst_to_src + self.bias_dst_to_src

        if self.self_feature_transform:
            x_self = self.lin_self(x)
            x_self = x_self + self.bias_self if self.use_bias else x_self
            return x_src_to_dst, x_dst_to_src, x_self
        else:
            return x_src_to_dst, x_dst_to_src


class FuzzyDirGCN(nn.Module):
    hidden_sizes: List[int]
    out_size: int
    alpha: float
    bias: bool
    self_feature_transform: bool
    self_loop: bool
    layer_wise_theta: bool
    use_activation: bool

    def setup(self):
        convs = []
        for hidden_size in self.hidden_sizes:
            convs.append(FuzzyDirGCNConv(hidden_size, use_bias=self.bias, self_feature_transform=self.self_feature_transform)) 
        self.convs = convs
        self.readout = nn.Dense(self.out_size, kernel_init=glorot_uniform())

    def __call__(self, x, edge_index, theta):
        if not self.layer_wise_theta:
            edge_index_frozen, edge_weight_frozen = compute_fuzzy_laplacian(
                edge_index, theta, x.shape[0], None, self.self_loop)

        for i, conv in enumerate(self.convs):
            if self.layer_wise_theta:
                conv_edge_index, conv_edge_weight = compute_fuzzy_laplacian(
                    edge_index, theta[i], x.shape[0], None, self.self_loop)
            else:
                conv_edge_index = edge_index_frozen
                conv_edge_weight = edge_weight_frozen

            x = conv(x, conv_edge_index, conv_edge_weight)

            if self.self_feature_transform:
                x_src_to_dst, x_dst_to_src, x_self = x
                x = self.alpha * x_src_to_dst + (1-self.alpha) * x_dst_to_src + x_self
            else:
                x_src_to_dst, x_dst_to_src = x
                x = self.alpha * x_src_to_dst + (1-self.alpha) * x_dst_to_src           
            if self.use_activation:
                x = nn.relu(x)

        x = self.readout(x)
            
        return x           