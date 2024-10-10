from typing import Optional

import torch
from torch_geometric.utils import remove_self_loops


def get_fuzzy_laplacian(
    edge_index: torch.Tensor,
    theta: torch.Tensor, 
    num_nodes: int,
    num_edges: int,
    edge_weight: Optional[torch.Tensor] = None,
    add_self_loop: Optional[bool] = False,
):
    
    assert num_edges == theta.size(0)
    if edge_weight is not None:
        assert num_edges == edge_weight.size(0)

    if edge_weight is not None:
        edge_weight = edge_weight
    else:
        edge_weight = torch.ones(edge_index.size(1), dtype=theta.dtype)
        edge_weight.to(theta.device)

    senders, receivers = edge_index[:, :num_edges]  
    conv_senders = torch.cat((senders, receivers))
    conv_receivers = torch.cat((receivers, senders))

    edge_director_src_to_tgt = torch.exp(1j * theta)
    edge_director_tgt_to_src = torch.exp(1j * (torch.pi / 2 - theta))
    edge_director = torch.cat((edge_director_src_to_tgt, edge_director_tgt_to_src))
    edge_weight = torch.cat((edge_weight, edge_weight))

    if add_self_loop:
        self_loops = torch.arange(num_nodes).to(conv_senders.device)
        conv_senders = torch.cat((conv_senders, self_loops))
        conv_receivers = torch.cat((conv_receivers, self_loops))
        edge_weight = torch.cat((edge_weight, torch.ones(num_nodes).to(conv_senders.device)))
        edge_director = torch.cat((edge_director, torch.full((num_nodes,), 1 + 1j).to(conv_senders.device)))

    out_weight = edge_director.real**2 * edge_weight
    in_weight = edge_director.imag**2 * edge_weight

    # conv_senders contain all i->j and i<-j edge indices
    deg_senders = torch.zeros(num_nodes, dtype=out_weight.dtype, device=theta.device) + 1e-12
    deg_senders.scatter_add_(0, conv_senders, out_weight) 
    deg_inv_sqrt_senders = torch.where(deg_senders<1e-11, 0.0, torch.rsqrt(deg_senders))

    deg_receivers = torch.zeros(num_nodes, dtype=out_weight.dtype, device=theta.device) + 1e-12
    deg_receivers.scatter_add_(0, conv_senders, in_weight) 
    deg_inv_sqrt_receivers = torch.where(deg_receivers<1e-11, 0.0, torch.rsqrt(deg_receivers))

    edge_weight_src_to_tgt = deg_inv_sqrt_senders[conv_senders] * out_weight * deg_inv_sqrt_receivers[conv_receivers]
    edge_weight_tgt_to_src = deg_inv_sqrt_receivers[conv_senders] * in_weight * deg_inv_sqrt_senders[conv_receivers]   

    # concatenate indices and weights of edges to process PyTorch Geometric's batching
    num_repeat = edge_index.shape[1] // num_edges
    conv_senders_batch, conv_receivers_batch = [conv_senders], [conv_receivers]
    for n in range(1, num_repeat):
        # increment node index
        conv_senders_batch.append(num_nodes * n + conv_senders)  
        conv_receivers_batch.append(num_nodes * n + conv_receivers)
    conv_senders_batch = torch.cat(conv_senders_batch)
    conv_receivers_batch = torch.cat(conv_receivers_batch)

    edge_weight_src_to_tgt_batch = edge_weight_src_to_tgt.repeat(num_repeat).unsqueeze(-1)
    edge_weight_tgt_to_src_batch = edge_weight_tgt_to_src.repeat(num_repeat).unsqueeze(-1)

    conv_edge_index = torch.stack((conv_senders_batch, conv_receivers_batch), dim=0)
    conv_edge_weight = (edge_weight_src_to_tgt_batch, edge_weight_tgt_to_src_batch)
    
    return conv_edge_index, conv_edge_weight