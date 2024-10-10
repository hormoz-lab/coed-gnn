import yaml
import numpy as np
import torch


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def use_best_hyperparams(args, dataset, model_name, experiment_type):
    best_params_file = f"best_hyperparams_{experiment_type}.yaml"
    with open(best_params_file, "r") as file:
        hyperparams = yaml.safe_load(file)

    for key, value in hyperparams[model_name][dataset].items():
        if hasattr(args, key):
            value = None if value == "None" else value
            setattr(args, key, value)

    return args


def get_edge_index_and_theta(adj):
    if isinstance(adj, torch.Tensor):
        A = adj.cpu().numpy()
    elif isinstance(adj, np.ndarray):
        A = adj
    triu_edges = np.array(np.triu_indices(len(A))).T[A[np.triu_indices(len(A))]!=0]
    tril_edges = np.array(np.tril_indices(len(A))).T[A[np.tril_indices(len(A))]!=0]
    tril_edges_flip = np.copy(tril_edges)
    tril_edges_flip[:, [0, 1]] = tril_edges_flip[:, [1, 0]]
    
    triu_edges_set = set(tuple(el) for el in triu_edges)
    tril_edges_set = set(tuple(el) for el in tril_edges)
    tril_edges_flip_set = set(tuple(el) for el in tril_edges_flip) 
    
    triu_symm_edges_set = triu_edges_set & tril_edges_flip_set
    tril_symm_edges_set = set(el[::-1] for el in triu_symm_edges_set)
    
    triu_dir_edges_set = triu_edges_set - triu_symm_edges_set
    tril_dir_edges_set = tril_edges_set - tril_symm_edges_set
    
    triu_dir_edges = sorted(list(triu_dir_edges_set))
    tril_dir_edges = sorted(list(tril_dir_edges_set))
    triu_symm_edges = sorted(list(triu_symm_edges_set))
    
    if len(triu_symm_edges) > 0:
        if len(triu_dir_edges)==0 and len(tril_dir_edges)==0:
            processed_edges = np.array(triu_symm_edges)
            theta = [np.pi/4] * len(triu_symm_edges) 
        else:
            processed_edges = np.vstack((triu_dir_edges, tril_dir_edges, triu_symm_edges))
            theta = [0] * (len(triu_dir_edges) + len(tril_dir_edges)) + [np.pi/4] * len(triu_symm_edges) 
    else:
        processed_edges = np.vstack((triu_dir_edges, tril_dir_edges))
        theta = [0] * (len(triu_dir_edges) + len(tril_dir_edges))
        
    theta = np.array(theta)
    theta = torch.tensor(theta)
    true_theta = theta
    
    edge_index_fuzzy = torch.from_numpy(processed_edges.T)
    edge_index_fuzzy_reverse = torch.stack(tuple(edge_index_fuzzy)[::-1])

    if isinstance(adj, torch.Tensor):
        return edge_index_fuzzy.to(adj.device), edge_index_fuzzy_reverse.to(adj.device), theta.to(adj.device)
    elif isinstance(adj, np.ndarray):
        return edge_index_fuzzy, edge_index_fuzzy_reverse, theta