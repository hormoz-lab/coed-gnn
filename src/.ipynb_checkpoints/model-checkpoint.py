from typing import Optional
from collections import defaultdict

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F

from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn import MessagePassing, JumpingKnowledge, global_mean_pool
from torch_geometric.nn import MLP, GINEConv, GraphConv
from torch_geometric.nn import HeteroConv


from utils import get_fuzzy_laplacian


class FuzzyDirGCNConv(MessagePassing):
    """
    Directional message-passing layer with continuous edge directions.
    Supports separate incoming and outgoing convolutions, with optional 
    self-feature transformations.

    Uses edge weights derived from a fuzzy Laplacian, supplied externally.

    Returns:
        Tuple[Tensor, Tensor] or Tuple[Tensor, Tensor, Tensor]: 
        - x_src_to_dst: Aggregated in-neighbor features 
        - x_dst_to_src: Aggregated out-neighbor features
        - x_self (optional): Self-features if self-feature_transformation=True
    """
    def __init__(
        self, 
        in_channels, 
        out_channels, 
        bias=True, 
        aggr_method="add", 
        self_feature_transform=False,
        dtype=torch.float,
    ):
        super(FuzzyDirGCNConv, self).__init__(aggr=aggr_method)
        self.aggr = aggr_method 
        self.self_feature_transform = self_feature_transform 

        self.lin_src_to_dst = Linear(
            in_channels, out_channels, 
            bias=False, weight_initializer='glorot')

        self.lin_dst_to_src = Linear(
            in_channels, out_channels, 
            bias=False, weight_initializer='glorot')

        if self.self_feature_transform:
            self.lin_self = Linear(
                in_channels, out_channels, 
                bias=False, weight_initializer='glorot')
        else:
            self.lin_self = None

        if bias:
            self.bias_src_to_dst = Parameter(torch.empty(out_channels))
            self.bias_dst_to_src = Parameter(torch.empty(out_channels))       
            if self.self_feature_transform:
                self.bias_self = Parameter(torch.empty(out_channels))
            else:
                self.bias_self = None
        else:
            self.bias_src_to_dst = None
            self.bias_dst_to_src = None
            self.bias_self = None

    def reset_parameters(self):
        glorot(self.lin_src_to_dst)
        glorot(self.lin_dst_to_src)
        zeros(self.bias_src_to_dst)
        zeros(self.bias_dst_to_src)
        if self.lin_self is not None:
            glorot(self.lin_dst_to_src)
        if self.bias_self is not None:
            zeros(self.bias_self)

    def forward(self, x, edge_index, edge_weight):
        num_nodes = x.size(0)
        edge_weight_src_to_tgt, edge_weight_tgt_to_src = edge_weight

        x_src_to_dst = self.propagate(
            edge_index, x=x, edge_weight=edge_weight_src_to_tgt)
        x_dst_to_src = self.propagate(
            edge_index, x=x, edge_weight=edge_weight_tgt_to_src)

        x_src_to_dst = self.lin_src_to_dst(x_src_to_dst)
        x_dst_to_src = self.lin_dst_to_src(x_dst_to_src)

        if self.bias_src_to_dst is not None:
            x_src_to_dst = x_src_to_dst + self.bias_src_to_dst
            x_dst_to_src = x_dst_to_src + self.bias_dst_to_src

        if self.self_feature_transform:
            x_self = self.lin_self(x)
            if self.bias_self is not None:
                x_self = x_self + self.bias_self
            return x_src_to_dst, x_dst_to_src, x_self
        else:
            return x_src_to_dst, x_dst_to_src

    def message(self, x_j, edge_weight):
        return x_j * edge_weight.view(-1, 1)


class FuzzyDirGCNLayer(nn.Module):
    """
    Identical as FuzzyDirGCNConv except that
    """
    def __init__(
        self, 
        num_nodes, 
        num_edges, 
        in_channels, 
        out_channels, 
        alpha=None,
        bias=True, 
        aggr_method='add',
        self_feature_transform=False,
        self_loop=False, 
    ):     
        super().__init__()
        self.alpha = alpha
        
        self.get_fuzzy_laplacian = lambda e, t, w: get_fuzzy_laplacian(
            e, t, num_nodes, num_edges, w, self_loop)
        
        self.conv = FuzzyDirGCNConv(
            in_channels, out_channels, bias=bias, self_feature_transform=self_feature_transform)
        self.conv.reset_parameters()
        
    def forward(self, x, edge_index, theta, edge_weight=None):
        conv_edge_index, conv_edge_weight = self.get_fuzzy_laplacian(
            edge_index, theta, edge_weight) 
        xs = self.conv(x, conv_edge_index, conv_edge_weight)

        if len(xs) == 3:
            x_src_to_dst, x_dst_to_src, x_self = xs
            if self.alpha is not None:
                return self.alpha * x_src_to_dst + (1 - self.alpha ) * x_dst_to_src + x_self
            else:
                return x_src_to_dst + x_dst_to_src + x_self
        elif len(xs) == 2:
            x_src_to_dst, x_dst_to_src = xs
            if self.alpha is not None:
                return self.alpha * x_src_to_dst + (1 - self.alpha ) 
            else:
                return x_src_to_dst + x_dst_to_src 

      
class FuzzyDirGCN(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers,
        num_nodes,
        num_edges,
        alpha=None,
        bias=True,
        self_feature_transform=True,
        self_loop=False,
        layer_wise_theta=False,
        normalize=True,
        jumping_knowledge=None,
        regression=True,
        dropout_rate=0.0,
        dtype=torch.float,
    ):
        super().__init__()
        self.alpha = alpha
        self.self_loop = self_loop
        self.layer_wise_theta = layer_wise_theta
        self.normalize = normalize
        self.jumping_knowledge = jumping_knowledge
        self.regression=regression
        self.num_nodes = num_nodes
        self.dropout_rate = dropout_rate

        self.get_fuzzy_laplacian = lambda e, t, w: get_fuzzy_laplacian(
            e, t, num_nodes, num_edges, w, self_loop)
        
        self.convs = torch.nn.ModuleList()      
        for _ in range(num_layers):
            self.convs.append(
                FuzzyDirGCNConv(
                    in_channels, hidden_channels, 
                    bias=bias, self_feature_transform=self_feature_transform, 
                    dtype=dtype))
            in_channels = hidden_channels
        
        self.readout = Linear(
            hidden_channels, out_channels, 
            bias=bias, weight_initializer='glorot')

        if jumping_knowledge is not None:
            if jumping_knowledge == "cat":
                input_dim = hidden_channels * num_layers 
            else:
                input_dim = hidden_channels
            self.lin = Linear(input_dim, out_channels)
            self.jump = JumpingKnowledge(
                jumping_knowledge, hidden_channels, num_layers)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.readout.reset_parameters()

    def forward(self, x, edge_index, theta, edge_weight=None, batch=None):
        num_nodes = self.num_nodes

        if not self.layer_wise_theta:
            edge_index_frozen, edge_weight_frozen = self.get_fuzzy_laplacian(
                edge_index, theta, edge_weight)

        x_intermediate = [] # for jumping knowledge

        for i, conv in enumerate(self.convs):
            if self.layer_wise_theta:
                conv_edge_index, conv_edge_weight = self.get_fuzzy_laplacian(
                    edge_index, theta[i], edge_weight) 
            else:
                conv_edge_index = edge_index_frozen
                conv_edge_weight = edge_weight_frozen

            xs = conv(x, conv_edge_index, conv_edge_weight)
            if len(xs) == 3:
                x_src_to_dst, x_dst_to_src, x_self = xs
                if self.alpha is not None:
                    x = self.alpha * x_src_to_dst + (1 - self.alpha) * x_dst_to_src + x_self
                else:
                    x = x_src_to_dst + x_dst_to_src + x_self
            elif len(xs) == 2:
                x_src_to_dst, x_dst_to_src = xs
                if self.alpha is not None:
                    x = self.alpha * x_src_to_dst + (1 - self.alpha) * x_dst_to_src
                else:
                    x = x_src_to_dst + x_dst_to_src

            if i != len(self.convs) -1 or self.jumping_knowledge is not None:
                x = F.relu(x)
                if self.dropout_rate != 0:
                    x = F.dropout(x, p=self.dropout_rate, training=self.training)
                if self.normalize:
                    x = F.normalize(x, p=2, dim=1)

                x_intermediate.append(x)

        if self.jumping_knowledge is not None:
            x = self.jump(x_intermediate)
            x = self.lin(x)
        else:
            x = self.readout(x)
            
        if self.regression:
            return x
        else:
            return F.log_softmax(x, dim=-1)     


class GridNet(nn.Module):
    def __init__(
        self, 
        hidden_channels, 
        out_channels, 
        num_layers, 
        alpha,
        metadata, 
        num_bus_nodes, 
        num_ac_line_edges, 
        num_transformer_edges, 
        dim_ac_line_edges, 
        dim_transformer_edges, 
        layer_wise_theta,
        self_feature_transform,
        self_loop,
    ):
        super().__init__()
        self.metadata = metadata
        self.layer_wise_theta = layer_wise_theta

        ## 1. Encoding
        # node 
        self.node_encoder = nn.ModuleDict({
            node_type: Linear(-1, hidden_channels)
            for node_type in metadata[0]
        })

        # edge
        self.edge_encoder = nn.ModuleDict({
            '-'.join(edge_type): Linear(-1, hidden_channels)
            for edge_type in [('bus', 'ac_line', 'bus'), 
                              ('bus', 'transformer', 'bus')]
        })

        # incorporate edge feature into node feature
        self.merge_edge_with_node_transformer = GINEConv(
            MLP([hidden_channels, hidden_channels]), edge_dim=dim_transformer_edges)
        self.merge_edge_with_node_ac_line = GINEConv(
            MLP([hidden_channels, hidden_channels]), edge_dim=dim_ac_line_edges)

        ## 2. Message passing 
        # bring subnode features into bus feature
        self.initial_conv = HeteroConv({
            ('generator', 'generator_link', 'bus'): GraphConv(hidden_channels, hidden_channels),
            ('load', 'load_link', 'bus'): GraphConv(hidden_channels, hidden_channels),
            ('shunt', 'shunt_link', 'bus'): GraphConv(hidden_channels, hidden_channels),
        })

        # bus-to-bus message passing
        self.intermediate_convs = torch.nn.ModuleList()
        for l in range(num_layers):  
            internal_mp_conv_ac = FuzzyDirGCNLayer(
                num_bus_nodes, num_ac_line_edges, hidden_channels, hidden_channels, alpha, 
                self_feature_transform=self_feature_transform, self_loop=self_loop)
            internal_mp_conv_trans = FuzzyDirGCNLayer(
                num_bus_nodes, num_transformer_edges, hidden_channels, hidden_channels, alpha,
                self_feature_transform=self_feature_transform, self_loop=self_loop)
                    
            conv = HeteroConv({
                ('bus', 'ac_line', 'bus'): internal_mp_conv_ac,  
                ('bus', 'transformer', 'bus'): internal_mp_conv_trans,  
            })
            self.intermediate_convs.append(conv)

        # pass bus feature to generator subnode
        self.final_conv = HeteroConv({
            ('bus', 'generator_link', 'generator'): GraphConv(hidden_channels, hidden_channels),
        })

        ## 3. Decoding
        self.generator_readout = Linear(hidden_channels, out_channels)
        

    def forward(
        self, 
        node_dict, 
        edge_index_dict, 
        edge_attr_dict, 
        fuzzy_theta_dict, 
        fuzzy_edge_weight_dict,
    ):
        x_dict = defaultdict(torch.Tensor)

        ## 1. Encoding
        for node_type, layer in self.node_encoder.items():
            x_dict[node_type] = layer(node_dict[node_type])

        x_transformer_merged = self.merge_edge_with_node_transformer(
            x=x_dict['bus'], 
            edge_index=edge_index_dict[('bus', 'transformer', 'bus')],
            edge_attr=edge_attr_dict[('bus', 'transformer', 'bus')],
        )
        x_ac_line_merged = self.merge_edge_with_node_ac_line(
            x=x_dict['bus'], 
            edge_index=edge_index_dict[('bus', 'ac_line', 'bus')],
            edge_attr=edge_attr_dict[('bus', 'ac_line', 'bus')]
        )    
        x_dict['bus'] = x_transformer_merged + x_ac_line_merged

        _x_dict = self.initial_conv(x_dict, edge_index_dict)

        ## 2. Message passing
        x_real, x_imag = _x_dict['bus'], _x_dict['bus']
        for l, conv in enumerate(self.intermediate_convs):
            if self.layer_wise_theta:
                _x_dict = conv(
                    _x_dict, edge_index_dict, 
                    theta_dict=fuzzy_theta_dict[l], 
                    edge_weight_dict=fuzzy_edge_weight_dict)
            else:
                _x_dict = conv(
                    _x_dict, edge_index_dict, 
                    theta_dict=fuzzy_theta_dict, 
                    edge_weight_dict=fuzzy_edge_weight_dict)

            _x_dict = {key: x.relu() for key, x in _x_dict.items()}

        _x_dict['generator'] = torch.zeros_like(x_dict['generator'])
        __x_dict = self.final_conv(_x_dict, edge_index_dict)

        ## 3. Decoding
        x_gen = self.generator_readout(__x_dict['generator'])
        return x_gen