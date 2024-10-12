import sys
import os
from functools import partial 
sys.path.append(os.path.abspath(os.path.join('..', 'src')))

import pickle
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrandom
import flax.linen as nn
import optax

from utils import get_graph_ensemble_dataset, use_best_hyperparams
from utils.arguments import args
from utils import JaxFuzzyDirGCN


def _loss_fuzzy(
    model, 
    params, 
    train_dataset, 
    edge_index, 
    theta,
):
    def _model_eval(carry, snapshot):
        cost = carry
        x, y = snapshot
        y_hats = jax.vmap(
            model, 
            in_axes=(None, 0, None, None),
        )(params, x.T[..., jnp.newaxis], edge_index, theta).T.squeeze()
        y_combined = jnp.hstack(
            (x[:, 1:], y[:, jnp.newaxis]))
        cost = cost + jnp.mean((y_hats - y_combined)**2)
        return cost, None
    
    cost, _ = jax.lax.scan(_model_eval, 0.0, train_dataset)
    return cost / train_dataset[0].shape[0]

    
@partial(jax.jit, static_argnums=0)
def _loss_fuzzy_test(
    model, 
    params, 
    test_dataset, 
    edge_index, 
    theta,
):
    def _model_eval(carry, snapshot):
        cost = carry
        x, y = snapshot
        y_hats = jax.vmap(
            model, 
            in_axes=(None, 0, None, None),
        )(params, x.T[..., jnp.newaxis], edge_index, theta).T.squeeze()[:, -1]
        y_combined = jnp.hstack(
            (x[:, 1:], y[:, jnp.newaxis]))[:, -1]
        cost = cost + jnp.mean((y_hats - y_combined)**2)
        return cost, None
    
    cost, _ = jax.lax.scan(_model_eval, 0.0, test_dataset)
    return cost / test_dataset[0].shape[0]


@partial(jax.jit, static_argnums=(0, 2, 4))
def train_step(
    model, 
    params, 
    opt, 
    opt_state, 
    theta_opt, 
    theta_opt_state, 
    train_dataset, 
    edge_index, 
    theta,
):
    loss_val, (param_grad, theta_grad) = jax.value_and_grad(
        _loss_fuzzy, 
        argnums=(1, 4))(model, params, train_dataset, edge_index, theta)
    updates, opt_state = opt.update(param_grad, opt_state, params)
    params = optax.apply_updates(params, updates)  
    
    theta_updates, theat_opt_state = theta_opt.update(theta_grad, theta_opt_state, theta)
    theta = optax.apply_updates(theta, theta_updates)
    theta = jax.lax.clamp(0.0, theta, jnp.pi/2)  
    return loss_val, params, opt_state, theta, theta_opt_state


def _loss_fuzzy_layer_wise(
    model, 
    params, 
    train_dataset, 
    edge_index, 
    theta1, 
    theta2,
):
    def _model_eval(carry, snapshot):
        cost = carry
        x, y = snapshot
        y_hats = jax.vmap(
            model, 
            in_axes=(None, 0, None, None),
        )(params, x.T[..., jnp.newaxis], edge_index, (theta1, theta2)).T.squeeze()
        y_combined = jnp.hstack(
            (x[:, 1:], y[:, jnp.newaxis]))
        cost = cost + jnp.mean((y_hats - y_combined)**2)
        return cost, None
    
    cost, _ = jax.lax.scan(_model_eval, 0.0, train_dataset)
    return cost / train_dataset[0].shape[0]


@partial(jax.jit, static_argnums=0)
def _loss_fuzzy_test_layer_wise(
    model, 
    params, 
    test_dataset, 
    edge_index, 
    theta1,
    theta2,
):
    def _model_eval(carry, snapshot):
        cost = carry
        x, y = snapshot
        y_hats = jax.vmap(
            model, 
            in_axes=(None, 0, None, None),
        )(params, x.T[..., jnp.newaxis], edge_index, (theta1, theta2)).T.squeeze()[:, -1]
        y_combined = jnp.hstack(
            (x[:, 1:], y[:, jnp.newaxis]))[:, -1]
        cost = cost + jnp.mean((y_hats - y_combined)**2)
        return cost, None
    
    cost, _ = jax.lax.scan(_model_eval, 0.0, test_dataset)
    return cost / test_dataset[0].shape[0]


@partial(jax.jit, static_argnums=(0, 2, 4, 5))
def train_step_layer_wise(
    model, 
    params,
    opt, 
    opt_state,
    theta_opt1, 
    theta_opt2, 
    theta_opt_state1, 
    theta_opt_state2, 
    train_dataset, 
    edge_index, 
    theta1, 
    theta2,
):
    loss_val, (param_grad, theta1_grad, theta2_grad) = jax.value_and_grad(
        _loss_fuzzy_layer_wise, 
        argnums=(1, 4, 5))(model, params, train_dataset, edge_index, theta1, theta2)
    updates, opt_state = opt.update(param_grad, opt_state, params)
    params = optax.apply_updates(params, updates)  

    new_theta = []
    new_theta_opt_state = []
    for _opt, _opt_state, _grad, _theta in zip([theta_opt1, theta_opt2], 
                                               [theta_opt_state1, theta_opt_state2],
                                               [theta1_grad, theta2_grad], 
                                               [theta1, theta2]):
        theta_updates, theta_opt_state = _opt.update(_grad, _opt_state, _theta)
        _tmp_theta = optax.apply_updates(_theta, theta_updates)
        _tmp_theta = jax.lax.clamp(0.0, _tmp_theta, jnp.pi/2)  
        new_theta.append(_tmp_theta)
        new_theta_opt_state.append(theta_opt_state)
    return loss_val, params, opt_state, new_theta, new_theta_opt_state


def run(args):
    gpus = jax.devices('gpu')
    with jax.default_device(gpus[args.gpu_idx]):
        random_key = jrandom.key(42)

        train_dataset, test_dataset, edge_index = get_graph_ensemble_dataset(args.dataset)

        n_repeats = 7
        test_losses = []
        for _ in range(n_repeats):
            seed_key, random_key = jrandom.split(random_key)
            model = JaxFuzzyDirGCN(
                hidden_sizes=[args.hidden_dimension] * args.num_layers,
                out_size=1,
                alpha=args.alpha,
                bias=True,
                self_feature_transform=args.self_feature_transform,
                self_loop=args.self_loop,
                layer_wise_theta=args.layer_wise_theta,
                use_activation=True)
            
            default_value = np.pi/4
            if args.layer_wise_theta:
                theta1 = jnp.array([np.pi/4] * edge_index.shape[-1])
                theta2 = jnp.array([np.pi/4] * edge_index.shape[-1])
                theta_opt1 = optax.adam(learning_rate=args.theta_learning_rate)
                theta_opt2 = optax.adam(learning_rate=args.theta_learning_rate)
                theta_opt_state1 = theta_opt1.init(theta1)
                theta_opt_state2 = theta_opt2.init(theta2)
            else:
                theta = jnp.array([np.pi/4] * edge_index.shape[-1])
                theta_opt = optax.adam(learning_rate=args.theta_learning_rate) 
                theta_opt_state = theta_opt.init(theta)

            _theta = (theta1, theta2) if args.layer_wise_theta else theta
            params = model.init(
                seed_key, train_dataset[0][0][:, 0][:, jnp.newaxis], edge_index, _theta)
            opt = optax.adam(learning_rate=args.learning_rate)
            opt_state = opt.init(params)
    
            best_test_loss = np.inf
            num_nondecreasing_step = 0
    
            for epoch in range(1, 5000):
                if args.layer_wise_theta:
                    (loss_tr, params, opt_state, 
                     (theta1, theta2), 
                     (theta_opt_state1, theta_opt_state2)) = train_step_layer_wise(
                        model.apply, 
                        params,
                        opt, opt_state, 
                        theta_opt1, theta_opt2, theta_opt_state1, theta_opt_state2,
                        train_dataset,
                        edge_index, 
                        theta1, theta2)
                    loss_test = _loss_fuzzy_test_layer_wise(
                        model.apply, params, test_dataset, edge_index, theta1, theta2)
                else:        
                    (loss_tr, params, opt_state, 
                     theta, 
                     theta_opt_state) = train_step(
                        model.apply, 
                        params,
                        opt, opt_state, 
                        theta_opt, theta_opt_state,
                        train_dataset,
                        edge_index, 
                        theta)
                    loss_test = _loss_fuzzy_test(
                        model.apply, params, test_dataset, edge_index, theta)
            
                if loss_test.item() < best_test_loss:
                    best_test_loss = loss_test.item()
                    num_nondecreasing_step = 0
                else:
                    num_nondecreasing_step += 1
        
                if num_nondecreasing_step > args.patience:
                    break
            
                if epoch % args.print_interval == 0:
                    print(f'epoch: {epoch}, tr/test loss: {loss_tr.item():.6f}/{loss_test.item():.6f}, '
                          f'# non-decreasing steps: {num_nondecreasing_step}')
            
            print(f'best test loss: {best_test_loss:.6f}\n')
            test_losses.append(best_test_loss)
    
        top_5_idx = np.argsort(test_losses)[:5]
        top_5 = [test_losses[i] for i in top_5_idx]
        print(f'top 5 test MSE: {np.mean(top_5)*100:.6f} +/- {np.std(top_5)*100:.6f}')
    
    
if __name__ == "__main__":
    args = use_best_hyperparams(args, "web_traffic", args.model, "real_ensemble") if args.use_best_hyperparams else args
    run(args)