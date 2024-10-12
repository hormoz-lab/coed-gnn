import sys
import os
sys.path.append(os.path.abspath(os.path.join('..', 'src')))

import pickle
import numpy as np
import torch

from utils import get_graph_ensemble_dataset, set_seed, use_best_hyperparams, masked_regression_loss
from utils.arguments import args
from model import FuzzyDirGCN, GridNet


def train(model, optimizer, theta, train_loader, edge_weight):
    model.train()            
    total_loss = 0
    for n, batch in enumerate(train_loader):
        optimizer.zero_grad()
        pred = model(batch.x, batch.edge_index, theta, edge_weight)
        loss = masked_regression_loss(pred, batch.y, batch.mask)
        loss.backward()
        total_loss += loss.item() * batch.num_graphs
        optimizer.step()
        # keep phase angle in [0, pi/2]
        with torch.no_grad():
            if isinstance(theta, list):
                for t in theta:
                    t.clamp_(0, torch.pi/2)
            elif isinstance(theta, torch.Tensor):
                theta.clamp_(0, torch.pi/2)
    return total_loss / (train_loader.batch_size * (n+1))

@torch.no_grad()
def test(model, theta, loader, edge_weight):
    model.eval()
    total_error = 0
    for n, batch in enumerate(loader):
        pred = model(batch.x, batch.edge_index, theta, edge_weight)
        total_error += masked_regression_loss(pred, batch.y, batch.mask).item()
    return total_error / (n + 1)

def load_model(args, info, device, metadata=None):
    if args.dataset == 'perturb_seq':
        model = FuzzyDirGCN(
            in_channels=info['in_channels'], 
            hidden_channels=args.hidden_dimension, 
            out_channels=info['out_channels'], 
            num_layers=args.num_layers,
            num_nodes=info['num_nodes'],
            num_edges=info['num_edges'],
            alpha=args.alpha,
            normalize=args.normalize,
            self_feature_transform=args.self_feature_transform,
            self_loop=args.self_loop,
            layer_wise_theta=args.layer_wise_theta,
            regression=True,
            dropout_rate=args.dropout_rate,
            jumping_knowledge=args.jumping_knowledge).to(device) 
        model.reset_parameters()
        return model
        
    elif args.dataset == 'power_grid':
        return GridNet(
            hidden_channels=args.hidden_dimension,
            out_channels=info['out_channels'],
            num_layers=args.num_layers,
            metadata=metadata,
            num_bus_nodes=info['num_bus_nodes'], 
            num_ac_line_edges=info['num_ac_line_edges'], 
            num_transformer_edges=info['num_transformer_edges'], 
            dim_ac_line_edges=info['dim_ac_line_edges'], 
            dim_transformer_edges=info['dim_transformer_edges'], 
            layer_wise_theta=args.layer_wise_theta,
            self_feature_transform=args.self_feature_transform,
            self_loop=args.self_loop).to(device)
        return model


def run(args):
    device = torch.device(f'cuda:{args.gpu_idx}')

    train_loader, val_loader, test_loader, info = get_graph_ensemble_dataset(
        args.dataset, device=device, undirected=args.undirected)
    
    num_nodes = info['num_nodes']
    num_edges = info['num_edges']
    in_channels = info['in_channels']
    out_channels = info['out_channels']

    set_seed(42)

    n_repeats = 7
    test_losses = []
    for _ in range(n_repeats):
        model = FuzzyDirGCN(
            in_channels=info['in_channels'], 
            hidden_channels=args.hidden_dimension, 
            out_channels=info['out_channels'], 
            num_layers=args.num_layers,
            num_nodes=info['num_nodes'],
            num_edges=info['num_edges'],
            alpha=args.alpha,
            normalize=args.normalize,
            self_feature_transform=args.self_feature_transform,
            self_loop=args.self_loop,
            layer_wise_theta=args.layer_wise_theta,
            regression=True,
            dropout_rate=args.dropout_rate,
            jumping_knowledge=args.jumping_knowledge).to(device) 
        model.reset_parameters()
        
        if args.layer_wise_theta:
            theta = [
                torch.tensor([torch.pi/4]*num_edges, device=device, requires_grad=True) 
                for _ in range(args.num_layers)
            ]
        else:
            theta = torch.tensor([torch.pi/4] * num_edges, device=device, requires_grad=True)
        edge_weight = torch.ones(num_edges).to(device)
        param_groups = [
            {'params': model.parameters(), 'lr': args.learning_rate, 'weight_decay': args.weight_decay}, 
            {'params': theta, 'lr': args.theta_learning_rate},
        ] 
        optimizer = torch.optim.Adam(param_groups)

        best_val_loss = torch.inf
        num_nondecreasing_step = 0

        theta_traj = []
        for epoch in range(1, 5000):
            train_loss = train(model, optimizer, theta, train_loader, edge_weight)
            val_loss = test(model, theta, val_loader, edge_weight)
        
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_test_loss = test(model, theta, test_loader, edge_weight)
                num_nondecreasing_step = 0
            else:
                num_nondecreasing_step += 1
        
            if num_nondecreasing_step > args.patience:
                break
        
            if epoch % args.print_interval == 0:
                print(f'epoch: {epoch}, tr/val loss: {train_loss:.6f}/{val_loss:.6f}, '
                      f'# non-decreasing steps: {num_nondecreasing_step}')
        
        print(f'best test loss: {best_test_loss:.6f}\n')
        test_losses.append(best_test_loss)

    top_5_idx = np.argsort(test_losses)[:5]
    top_5 = [test_losses[i] for i in top_5_idx]
    print(f'top 5 test MSE: {np.mean(top_5)*100:.6f} +/- {np.std(top_5)*100:.6f}')


if __name__ == "__main__":
    args = use_best_hyperparams(args, "perturb_seq", args.model, "real_ensemble") if args.use_best_hyperparams else args
    run(args)