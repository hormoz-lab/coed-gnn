import sys
import os
sys.path.append(os.path.abspath(os.path.join('..', 'src')))

import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj

from utils import get_classification_dataset, get_edge_index_and_theta, set_seed, use_best_hyperparams
from utils.arguments import args
from model import FuzzyDirGCN


def train(x, y, model, optimizer, edge_index, theta, edge_weight, mask, index=None):
    model.train()
    optimizer.zero_grad()
    if index is not None:
        loss = F.nll_loss(
            model(x, edge_index, theta, edge_weight)[mask[:, index]], 
            y[mask[:, index]])
    else:
        loss = F.nll_loss( 
            model(x, edge_index, theta, edge_weight)[mask], 
            y[mask])  
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test(x, y, model, optimizer, edge_index, theta, edge_weight, masks, index):
    model.eval()
    log_probs, accs = model(x, edge_index, theta, edge_weight), []
    for mask in masks:
        if index is not None:
            pred = log_probs[mask[:, index]].max(1)[1]
            acc = pred.eq(y[mask[:, index]]).sum().item() / mask[:, index].sum().item()
        else:
            pred = log_probs[mask].max(1)[1]
            acc = pred.eq(y[mask]).sum().item() / mask.sum().item()            
        accs.append(acc)
    return accs


def run(args):
    device = torch.device(f'cuda:{args.gpu_idx}')

    data, (train_mask, val_mask, test_mask) = get_classification_dataset(
        args.dataset, device=device)

    adj = to_dense_adj(data.edge_index)[0]
    if args.remove_existing_self_loop:
        adj.fill_diagonal_(0.0)

    src_to_dst_edge, dst_to_src_edge, theta = get_edge_index_and_theta(adj)
    theta = theta.float()
    edge_index = src_to_dst_edge 
    edge_weight = torch.ones(edge_index.shape[1]).to(device)

    num_nodes = data.x.shape[0]
    num_edges = edge_index.shape[1]
    
    in_channels = data.x.shape[-1]
    out_channels = data.y.max().item() + 1

    set_seed(42)

    test_accs = []     
    for index in range(10):

        model = FuzzyDirGCN(
            in_channels=in_channels, 
            hidden_channels=args.hidden_dimension, 
            out_channels=out_channels, 
            num_layers=args.num_layers,
            num_nodes=num_nodes,
            num_edges=num_edges,
            alpha=args.alpha,
            normalize=args.normalize,
            self_feature_transform=args.self_feature_transform,
            self_loop=args.self_loop,
            layer_wise_theta=args.layer_wise_theta,
            regression=False,
            dropout_rate=args.dropout_rate,
            jumping_knowledge=args.jumping_knowledge).to(device) 
        model.reset_parameters()
    
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
        best_val_acc = 0.0
        best_test_acc = 0.0
        n_non_decreasing_step = 0
    
        for epoch in range(1, 5000):
            tr_loss = train(
                data.x, data.y, model, optimizer, edge_index, theta, edge_weight, train_mask, index)
            train_acc, val_acc, test_acc = test( 
                data.x, data.y, model, optimizer, edge_index, theta, edge_weight, (train_mask, val_mask, test_mask), index)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                n_non_decreasing_step = 0
            else:
                n_non_decreasing_step += 1
            
            if n_non_decreasing_step > args.patience:
                break
        
            if epoch % args.print_interval == 0:
                print(f'index: {index} | '
                      f'Epoch: {epoch:03d}, Loss: {tr_loss:.5f}, Train: {train_acc:.5f}, Val: {val_acc:.5f}, '
                      f'Best: {best_test_acc:.4}, early_stopping: {n_non_decreasing_step}')
    
        test_accs.append(best_test_acc)

    print(f'test acc: {np.mean(test_accs)*100:.5f} +/- {np.std(test_accs)*100:.5f}')


if __name__ == "__main__":
    if args.use_best_hyperparams:
        args = use_best_hyperparams(args, args.dataset, args.model, "classification") 
    run(args)
