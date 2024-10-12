from collections import defaultdict

import numpy as np
from scipy.spatial import Delaunay

import jax
import jax.numpy as jnp
from jax.ops import segment_sum


## Lattice generation
def create_triangular_lattice(num_points, lattice_width=1.0):
    lattice = []
    rows = int(np.sqrt(num_points))
    cols = int(np.ceil(num_points / rows))    
    for row in range(rows):
        for col in range(cols):
            x = col * 1.5 * lattice_width - 2.0
            y = row * np.sqrt(3) * lattice_width - 2.0           
            if col % 2 == 1:
                y += np.sqrt(3) * lattice_width / 2
            if -2 <= x and x <= 2 and -2 <= y and y <= 2:   
                lattice.append((x, y))            
    return np.array(lattice)


def is_equilateral_triangle(triangle, lattice):
    lengths = [
        np.linalg.norm(lattice[triangle[i]] - lattice[triangle[j]]) 
        for i, j in [(0, 1), (1, 2), (2, 0)]
    ]
    return all(np.isclose(lengths, lengths[0]))


def create_equilateral_triangular_lattice(triangular_lattice):
    tri = Delaunay(triangular_lattice)
    edges = set()
    for simplex in tri.simplices:
        if is_equilateral_triangle(simplex, triangular_lattice):
            edges.add(tuple(sorted((simplex[0], simplex[1]))))
            edges.add(tuple(sorted((simplex[1], simplex[2]))))
            edges.add(tuple(sorted((simplex[2], simplex[0]))))        
    return edges


## Feature generation
aggr = lambda x, index, num_nodes: segment_sum(data=x,
                                               segment_ids=index,
                                               num_segments=num_nodes,
                                               indices_are_sorted=False,
                                               unique_indices=False)


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
    
    deg_senders = segment_sum(out_weight, conv_senders, num_nodes) + 1e-12
    deg_receivers = segment_sum(in_weight, conv_senders, num_nodes) + 1e-12

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


def propagate_features_fuzzy(
    x, edge_index, theta, num_nodes, edge_weight, 
    self_feature_transform, self_loop, alpha, 
    kernels=None, use_nonlinearity=False,
):
    if kernels is not None:
        src_to_dst_kernel, dst_to_src_kernel, self_kernel = kernels
        src_to_dst_propagator = lambda x: jnp.einsum('jk,ik->ij', src_to_dst_kernel, x)
        dst_to_src_propagator = lambda x: jnp.einsum('jk,ik->ij', dst_to_src_kernel, x)
        self_transform = lambda x: jnp.einsum('jk,ik->ij', self_kernel, x) 
    else:
        src_to_dst_propagator = dst_to_src_propagator = self_transform = lambda x: x
        
        
    conv_edge_index, conv_edge_weight = compute_fuzzy_laplacian(
        edge_index, theta, num_nodes, edge_weight, self_loop)
    senders, receivers = conv_edge_index
    edge_weight_src_to_tgt, edge_weight_tgt_to_src = conv_edge_weight

    # using senders->receivers in both operations because graph is undir.
    x_src_to_dst = aggr(
        x[senders] * edge_weight_src_to_tgt,
        receivers,
        num_nodes)
    
    x_dst_to_src = aggr(
        x[senders] * edge_weight_tgt_to_src,
        receivers,
        num_nodes)
    
    new_x = alpha * src_to_dst_propagator(x_src_to_dst) + \
            (1-alpha) * dst_to_src_propagator(x_dst_to_src)
    if self_feature_transform:
        new_x += self_transform(x)
    return new_x


def generate_features_fuzzy(features, *args, K, kernels, use_nonlinearity):
    hop_features = defaultdict(np.array)
    features = np.copy(features)   
    hop_features[0] = features
    for k in range(1, K):
        features = propagate_features_fuzzy(features, *args, kernels, use_nonlinearity)
        hop_features[k] = features
        hop_features[k] /= jnp.linalg.norm(hop_features[k], axis=-1, keepdims=True)
    return hop_features


## Potential (and vector field) generation
def single_source_single_sink(x):
    a = jnp.array([1, -1])
    mu = jnp.array([[-1, 1],
                    [1, -1]])    
    P = jnp.array([
        [[1, 0],
         [0, 1]],
        [[1, 0],
         [0, 1]],
    ])  
    a1, a2 = a
    mu1, mu2 = mu
    P1, P2 = P
    V = a1 * jnp.exp(-(x-mu1) @ P1 @ (x-mu1)) + \
        a2 * jnp.exp(-(x-mu2) @ P2 @ (x-mu2))
    return V


def solenoidal_vectorfield(x):
    return np.array([np.sin(np.pi*x[0]) * np.cos(np.pi*x[1]), 
                     -np.cos(np.pi*x[0]) * np.sin(np.pi*x[1])])