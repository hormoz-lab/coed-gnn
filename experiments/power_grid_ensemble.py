import sys
import os
sys.path.append(os.path.abspath(os.path.join('..', 'src')))

import pickle
import numpy as np
import torch

from utils import get_graph_ensemble_dataset, set_seed, use_best_hyperparams, masked_regression_loss
from utils.arguments import args
from model import FuzzyDirGCN, GridNet


def train(model, optimizer, train_loader, theta_dict, edge_weight_dict):
    model.train()            
    total_loss = 0
    for n, batch in enumerate(train_loader):
        optimizer.zero_grad()
        pred = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict, theta_dict, edge_weight_dict)
        loss = masked_regression_loss(pred, batch['generator'].y)
        loss.backward()
        total_loss += loss.item() * batch.num_graphs
        optimizer.step()
        with torch.no_grad():
            if isinstance(theta_dict, dict):
                theta_dict[('bus', 'ac_line', 'bus')].clamp_(0, torch.pi/2)
                theta_dict[('bus', 'transformer', 'bus')].clamp_(0, torch.pi/2)
            elif isinstance(theta_dict, list):
                for j in range(len(theta_dict)):
                    theta_dict[j][('bus', 'ac_line', 'bus')].clamp_(0, torch.pi/2)
                    theta_dict[j][('bus', 'transformer', 'bus')].clamp_(0, torch.pi/2)
    return total_loss / (train_loader.batch_size * (n+1))


@torch.no_grad()
def test(model, loader, theta_dict, edge_weight_dict):
    model.eval()
    total_error = 0
    for n, batch in enumerate(loader):
        pred = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict, theta_dict, edge_weight_dict)
        total_error += masked_regression_loss(pred, batch['generator'].y).item()
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
        
    train_loader, val_loader, test_loader, info, metadata = get_graph_ensemble_dataset(
        args.dataset, args.dataset_directory, device=device, undirected=args.undirected)
    
    num_bus_nodes = info['num_bus_nodes']
    num_ac_line_edges = info['num_ac_line_edges']
    num_transformer_edges = info['num_transformer_edges']
    dim_ac_line_edges = info['dim_ac_line_edges']
    dim_transformer_edges = info['dim_transformer_edges']

    set_seed(42)

    n_repeats = 7
    test_losses = []
    for _ in range(n_repeats):
        model = GridNet(
            hidden_channels=args.hidden_dimension,
            out_channels=info['out_channels'],
            num_layers=args.num_layers,
            alpha=args.alpha,
            metadata=metadata,
            num_bus_nodes=info['num_bus_nodes'], 
            num_ac_line_edges=info['num_ac_line_edges'], 
            num_transformer_edges=info['num_transformer_edges'], 
            dim_ac_line_edges=info['dim_ac_line_edges'], 
            dim_transformer_edges=info['dim_transformer_edges'], 
            layer_wise_theta=args.layer_wise_theta,
            self_feature_transform=args.self_feature_transform,
            self_loop=args.self_loop).to(device)
        
        if args.layer_wise_theta:
            opt_theta_ac = [
                torch.tensor([torch.pi/4]*num_ac_line_edges, device=device, requires_grad=True) 
                for _ in range(args.num_layers)
            ]
            opt_theta_trans = [
                torch.tensor([torch.pi/4]*num_transformer_edges, device=device, requires_grad=True) 
                for _ in range(args.num_layers)
            ]
            theta_dict = [
                {('bus', 'ac_line', 'bus'): _opt_theta_ac, ('bus', 'transformer', 'bus'): _opt_theta_trans}
                for _opt_theta_ac, _opt_theta_trans in zip(opt_theta_ac, opt_theta_trans)
            ]
            
        else:
            opt_theta_ac = torch.tensor([torch.pi/4]*num_ac_line_edges, device=device, requires_grad=True)        
            opt_theta_trans = torch.tensor([torch.pi/4]*num_transformer_edges, device=device, requires_grad=True)
            theta_dict = {('bus', 'ac_line', 'bus'): opt_theta_ac, ('bus', 'transformer', 'bus'): opt_theta_trans}
            
        param_groups = [
            {'params': model.parameters(), 'lr': args.learning_rate, 'weight_decay': args.weight_decay},
            {'params': opt_theta_ac, 'lr': args.theta_learning_rate},
            {'params': opt_theta_trans, 'lr': args.theta_learning_rate},
        ]
        optimizer = torch.optim.Adam(param_groups)

        edge_weight_dict = {
            ('bus', 'ac_line', 'bus'): torch.ones(num_ac_line_edges).to(device),
            ('bus', 'transformer', 'bus'): torch.ones(num_transformer_edges).to(device),
        }


        best_val_loss = torch.inf
        num_nondecreasing_step = 0

        theta_traj = []
        for epoch in range(1, 5000):
            train_loss = train(model, optimizer, train_loader, theta_dict, edge_weight_dict)
            val_loss = test(model, val_loader, theta_dict, edge_weight_dict)
        
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_test_loss = test(model, test_loader, theta_dict, edge_weight_dict)
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
    print(vars(args))
    args = use_best_hyperparams(args, args.dataset, args.model, "real_ensemble") if args.use_best_hyperparams else args
    print(vars(args))
    run(args)