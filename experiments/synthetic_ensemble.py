import sys
import os
sys.path.append(os.path.abspath(os.path.join('..', 'src')))

import pickle
import numpy as np
import torch

from utils import get_graph_ensemble_dataset, set_seed, use_best_hyperparams, masked_regression_loss
from utils.arguments import args
from model import FuzzyDirGCN


def train(model, optimizer, theta, train_loader, edge_weight, dataset):
    model.train()            
    total_loss = 0
    for n, batch in enumerate(train_loader):
        optimizer.zero_grad()
        pred = model(batch.x, batch.edge_index, theta, edge_weight)
        if dataset=='lattice':
            mask = None
        elif dataset=='grn':
            mask = batch.mask
        loss = masked_regression_loss(pred, batch.y, mask)
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
def test(model, theta, loader, edge_weight, dataset):
    model.eval()
    total_error = 0
    for n, batch in enumerate(loader):
        pred = model(batch.x, batch.edge_index, theta, edge_weight)
        if dataset=='lattice':
            mask = None
        elif dataset=='grn':
            mask = batch.mask
        total_error += masked_regression_loss(pred, batch.y, mask).item()
    return total_error / (n + 1)


def run(args):
    device = torch.device(f'cuda:{args.gpu_idx}')

    if args.pe_type is not None:
        train_loader, val_loader, test_loader, info, pe = get_graph_ensemble_dataset(
            args.dataset, device=device, undirected=args.undirected, 
            pe_type=args.pe_type, pe_dim=args.pe_dim)
    else:
        train_loader, val_loader, test_loader, info = get_graph_ensemble_dataset(
            args.dataset, device=device, undirected=args.undirected)

    num_nodes = info['num_nodes']
    num_edges = info['num_edges']
    in_channels = info['in_channels']
    out_channels = info['out_channels']

    set_seed(42)

    if args.store_theta:
        theta_trajs = []
        
    n_repeats = 7
    test_losses = []
    for _ in range(n_repeats):
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
            regression=True,
            dropout_rate=args.dropout_rate,
            jumping_knowledge=args.jumping_knowledge).to(device) 
        model.reset_parameters()

        best_val_loss = torch.inf
        num_nondecreasing_step = 0

        if args.layer_wise_theta:
            theta = [
                torch.tensor([torch.pi/4]*num_edges, device=device, requires_grad=True) 
                for _ in range(args.num_layers)
            ]
        else:
            theta = torch.tensor([np.pi/4] * num_edges, device=device, requires_grad=True)
        edge_weight = torch.ones(num_edges).to(device)
        param_groups = [
            {'params': model.parameters(), 'lr': args.learning_rate, 'weight_decay': args.weight_decay}, 
            {'params': theta, 'lr': args.theta_learning_rate},
        ] 
        optimizer = torch.optim.Adam(param_groups)

        theta_traj = []
        for epoch in range(1, 5000):
            train_loss = train(model, optimizer, theta, train_loader, edge_weight, args.dataset)
            val_loss = test(model, theta, val_loader, edge_weight, args.dataset)
        
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_test_loss = test(model, theta, test_loader, edge_weight, args.dataset)
                num_nondecreasing_step = 0
            else:
                num_nondecreasing_step += 1
        
            if num_nondecreasing_step > args.patience:
                break
        
            if epoch % args.print_interval == 0:
                print(f'epoch: {epoch}, tr/val loss: {train_loss:.6f}/{val_loss:.6f}, '
                      f'# non-decreasing steps: {num_nondecreasing_step}')
                if args.store_theta:
                    if isinstance(theta, list): 
                        theta_traj.append([_theta.detach().cpu().numpy() for _theta in theta])
                    else: 
                        theta_traj.append(theta.detach().cpu().numpy())
        
        print(f'best test loss: {best_test_loss:.6f}\n')
        test_losses.append(best_test_loss)
        if args.store_theta:
            theta_trajs.append(theta_traj)

    top_5_idx = np.argsort(test_losses)[:5]
    top_5 = [test_losses[i] for i in top_5_idx]
    print(f'top 5 test MSE: {np.mean(top_5)*100:.6f} +/- {np.std(top_5)*100:.6f}')

    if args.store_theta:
        top_theta_traj = [theta_trajs[i] for i in top_5_idx]
        pickle.dump(top_theta_traj, open(f'{args.dataset}_top_theta.pkl', 'wb'))


if __name__ == "__main__":
    args = use_best_hyperparams(args, args.dataset, args.model, "synthetic_ensemble") if args.use_best_hyperparams else args
    run(args)