import os
import pickle
from collections import defaultdict

import torch
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import Planetoid, WikipediaNetwork, WebKB
from torch_geometric.utils import sort_edge_index, to_undirected
from torch_geometric.transforms import AddLaplacianEigenvectorPE

from .graph_ensemble_dataset import GraphEnsembleDataset
from .opf import OPFDataset
from .electrostatic_encoding import get_electrostatic_function_encoding

try:
    import jax
    import jax.numpy as jnp
    from torch_geometric_temporal.dataset import WikiMathsDatasetLoader
    from torch_geometric_temporal.signal import temporal_signal_split
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


ROOT_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'datasets')
DEVICE = torch.device('cpu')


def get_classification_dataset(name, root_dir=ROOT_DIR, device=DEVICE):
    if name in ["cora"]:
        data_path = os.path.join(ROOT_DIR, 'standard_datasets', 'Planetoid')
        dataset = Planetoid(data_path, name, split='geom-gcn', transform=T.NormalizeFeatures()).to(device)
        data = dataset[0]
        train_mask = data.train_mask.to(bool)
        val_mask = data.val_mask.to(bool)
        test_mask = data.test_mask.to(bool)

    elif name in ["squirrel", "chameleon"]:
        data_path = os.path.join(ROOT_DIR, 'standard_datasets', 'WikipediaNetwork')
        dataset = WikipediaNetwork(data_path, name).to(device)
        data = dataset[0]
        train_mask = data['train_mask']
        val_mask = data['val_mask']
        test_mask = data['test_mask']

    elif name in ["texas", "wisconsin"]:
        data_path = os.path.join(ROOT_DIR, 'standard_datasets', 'WebKB')
        dataset = WebKB(data_path, name, transform=T.NormalizeFeatures()).to(device)
        data = dataset[0]
        train_mask = data['train_mask']
        val_mask = data['val_mask']
        test_mask = data['test_mask']
    
    return data, (train_mask, val_mask, test_mask)
    

def get_graph_ensemble_dataset(name, root_dir=ROOT_DIR, device=DEVICE, undirected=False, **pe_attr):
    if name in ["lattice"]:
        data_path = os.path.join(root_dir, 
                                 'triangular_lattice_graph',
                                 'random_feature_on_gradient_vector_field.pkl')
        data = pickle.load(open(data_path, 'rb'))
        edge_index = torch.tensor(data['edge_index'], device=device)
        if undirected:
            edge_index = to_undirected(edge_index)
        edge_feature = torch.ones_like(edge_index[0], device=device)

        X_train, Y_train = (
            torch.tensor(data['train_data'][0], device=device),
            torch.tensor(data['train_data'][1], device=device),
        )
        X_val, Y_val = (
            torch.tensor(data['val_data'][0], device=device),
            torch.tensor(data['val_data'][1], device=device),
        )
        X_test, Y_test = (
            torch.tensor(data['test_data'][0], device=device),
            torch.tensor(data['test_data'][1], device=device),
        )        

        info = {
            'num_nodes': X_train.shape[1],
            'num_edges': edge_index.shape[1],
            'in_channels': X_train.shape[-1],
            'out_channels': Y_train.shape[-1],
        }

        train_batch_size = 16
        val_batch_size = X_val.shape[0]
        test_batch_size = X_test.shape[0]
        
        train_dataset = GraphEnsembleDataset(
            root=os.path.join(root_dir, 'triangular_lattice_graph'),
            x=X_train, 
            y=Y_train, 
            edge_index=edge_index)
        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, drop_last=True)

        val_dataset = GraphEnsembleDataset(
            root=os.path.join(root_dir, 'triangular_lattice_graph'), 
            x=X_val, 
            y=Y_val, 
            edge_index=edge_index)
        val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

        test_dataset = GraphEnsembleDataset(
            root=os.path.join(root_dir, 'triangular_lattice_graph'), 
            x=X_test, 
            y=Y_test, 
            edge_index=edge_index)
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

        if pe_attr:
            if pe_attr['pe_type'] == 'eigenvector':
                pe_dim = pe_attr['pe_dim']
                pe = AddLaplacianEigenvectorPE(k=pe_dim)(train_dataset[0]).laplacian_eigenvector_pe.to(device)
            elif pe_attr['pe_type'] == 'electrostatic':
                print('here')
                num_nodes = train_dataset[0].x.shape[0]
                pe = get_electrostatic_function_encoding(edge_index, num_nodes).to(device)
            return train_loader, val_loader, test_loader, info, pe
        else:
            return train_loader, val_loader, test_loader, info

        
    elif name in ["grn"]:
        data_path = os.path.join(root_dir,
                                 'gene_regulatory_network',
                                 'grn.pkl')
        data = pickle.load(open(data_path, 'rb'))
        edge_index = torch.tensor(data['edge_index'], device=device)
        if undirected:
            edge_index = to_undirected(edge_index)
        edge_feature = torch.ones_like(edge_index[0], device=device)

        edge_index = torch.tensor(data['edge_index'], device=device)
        theta = torch.tensor(data['theta'], device=device)
        edge_feature = torch.ones_like(edge_index[0], device=device)
    
        X_train, Y_train, mask_train = (
            torch.tensor(data['X_train'], device=device), 
            torch.tensor(data['Y_train'], device=device).unsqueeze(2),
            torch.tensor(data['pert_tracks_train'], device=device),
        )
        X_val, Y_val, mask_val = (
            torch.tensor(data['val_data']['X_double'], device=device),
            torch.tensor(data['val_data']['Y_double'], device=device).unsqueeze(2),
            torch.tensor(data['val_data']['pert_tracks_double'], device=device),
        )
        X_test, Y_test, mask_test = (
            torch.tensor(data['test_data']['X_double'], device=device), 
            torch.tensor(data['test_data']['Y_double'], device=device).unsqueeze(2),
            torch.tensor(data['test_data']['pert_tracks_double'], device=device),
        )

        info = {
            'num_nodes': X_train.shape[1],
            'num_edges': edge_index.shape[1],
            'in_channels': X_train.shape[-1],
            'out_channels': Y_train.shape[-1],
        }

        train_batch_size = 8
        val_batch_size = X_val.shape[0]
        test_batch_size = X_test.shape[0]

        train_dataset = GraphEnsembleDataset(
            root=os.path.join(root_dir, 'gene_regulatory_network'), 
            x=X_train, 
            y=Y_train, 
            edge_index=edge_index,
            mask=mask_train)
        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, drop_last=False)

        val_dataset = GraphEnsembleDataset(
            root=os.path.join(root_dir, 'gene_regulatory_network'), 
            x=X_val, 
            y=Y_val, 
            edge_index=edge_index,
            mask=mask_val)
        val_loader = DataLoader(val_dataset, batch_size=val_batch_size)

        test_dataset = GraphEnsembleDataset(
            root=os.path.join(root_dir, 'gene_regulatory_network'), 
            x=X_test, 
            y=Y_test, 
            edge_index=edge_index,
            mask=mask_test)
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size)

        if pe_attr:
            if pe_attr['pe_type'] == 'eigenvector':
                pe_dim = pe_attr['pe_dim']
                pe = AddLaplacianEigenvectorPE(k=pe_dim)(train_dataset[0]).laplacian_eigenvector_pe.to(device)
            elif pe_attr['pe_type'] == 'electrostatic':
                num_nodes = train_dataset[0].x.shape[0]
                pe = get_electrostatic_function_encoding(edge_index, num_nodes).to(device)
            return train_loader, val_loader, test_loader, info, pe
        else:
            return train_loader, val_loader, test_loader, info
        

    elif name in ["perturb_seq"]:
        data_path = os.path.join(root_dir,
                                 'perturb_seq',
                                 'Replogle-gwps.pkl')
        data = pickle.load(open(data_path, 'rb'))
        edge_index = torch.tensor(data['edge_index'], device=device)
        edge_feature = torch.ones_like(edge_index[0], device=device)

        X_train, Y_train, mask_train = data['train_data'].values()
        X_train, Y_train, mask_train = (
            torch.from_numpy(X_train).unsqueeze(-1).to(device), 
            torch.from_numpy(Y_train).unsqueeze(-1).to(device), 
            torch.from_numpy(mask_train).unsqueeze(-1).to(device),
        )
        X_val, Y_val, mask_val = data['val_data'].values()
        X_val, Y_val, mask_val = (
            torch.from_numpy(X_val).unsqueeze(-1).to(device), 
            torch.from_numpy(Y_val).unsqueeze(-1).to(device), 
            torch.from_numpy(mask_val).unsqueeze(-1).to(device),
        )
        X_test, Y_test, mask_test = data['test_data'].values()
        X_test, Y_test, mask_test = (
            torch.from_numpy(X_test).unsqueeze(-1).to(device), 
            torch.from_numpy(Y_test).unsqueeze(-1).to(device), 
            torch.from_numpy(mask_test).unsqueeze(-1).to(device),
        )

        info = {
            'num_nodes': X_train.shape[1],
            'num_edges': edge_index.shape[1],
            'in_channels': X_train.shape[-1],
            'out_channels': Y_train.shape[-1],
        }

        train_batch_size = 16
        val_batch_size = X_val.shape[0]
        test_batch_size = X_test.shape[0]
                    
        train_dataset = GraphEnsembleDataset(
            root=os.path.join(root_dir, 'perturb_seq'), 
            x=X_train, 
            y=Y_train, 
            edge_index=edge_index,
            mask=mask_train)
        train_loader = DataLoader(
            train_dataset, batch_size=train_batch_size, shuffle=False, drop_last=True)
        
        val_dataset = GraphEnsembleDataset(
            root=os.path.join(root_dir, 'perturb_seq'), 
            x=X_val, 
            y=Y_val, 
            edge_index=edge_index,
            mask=mask_val)
        val_loader = DataLoader(
            val_dataset, batch_size=val_batch_size, shuffle=False)  
        
        test_dataset = GraphEnsembleDataset(
            root=os.path.join(root_dir, 'perturb_seq'), 
            x=X_test, 
            y=Y_test, 
            edge_index=edge_index,
            mask=mask_test)
        test_loader = DataLoader(
            test_dataset, batch_size=test_batch_size, shuffle=False)   

        return train_loader, val_loader, test_loader, info  

    elif name in ["web_traffic"]:
        if JAX_AVAILABLE:
            loader = WikiMathsDatasetLoader()
            dataset = loader.get_dataset(lags=8) 
            _train_dataset, _test_dataset = temporal_signal_split(dataset, train_ratio=0.9)
            
            edge_index = jnp.array(next(iter(_train_dataset)).edge_index)

            train_dataset_jax = (
                jnp.stack([snapshot.x.numpy() for snapshot in _train_dataset]),
                jnp.stack([snapshot.y.numpy() for snapshot in _train_dataset]),
            )
            test_dataset_jax = (
                jnp.stack([snapshot.x.numpy() for snapshot in _test_dataset]),
                jnp.stack([snapshot.y.numpy() for snapshot in _test_dataset]),
            )
            return train_dataset_jax, test_dataset_jax, edge_index
        else:
            pass
        
    elif name in ["power_grid"]:
        data_path = os.path.join(root_dir, 'OPF')        
        dataset = OPFDataset(
            data_path, 
            case_name='pglib_opf_case2000_goc', 
            split='train', 
            num_groups=1,
        ).to(device)

        indices = list(range(len(dataset)))

        train_size = 200
        val_size = 50
        test_size = 50
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size+val_size]
        test_indices = indices[train_size+val_size:train_size+val_size+test_size]

        dataset_small = defaultdict(list)

        for data_type, indices in zip(['train', 'val', 'test'], 
                                         [train_indices, val_indices, test_indices]):
            for data in torch.utils.data.Subset(dataset, indices):
                _data = data.coalesce().clone()
                _data[('bus', 'ac_line', 'bus')].edge_index = sort_edge_index(
                    _data[('bus', 'ac_line', 'bus')].edge_index)
                _data[('bus', 'transformer', 'bus')].edge_index = sort_edge_index(
                    _data[('bus', 'transformer', 'bus')].edge_index)
                dataset_small[data_type].append(_data)

        train_batch_size = 16    
        test_batch_size = 50
        
        train_loader = DataLoader(
            dataset_small['train'], batch_size=train_batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(
            dataset_small['val'], batch_size=test_batch_size, shuffle=False, drop_last=False)
        test_loader = DataLoader(
            dataset_small['test'], batch_size=test_batch_size, shuffle=False, drop_last=False)

        _datum =  dataset_small['train'][0]
        info = {
            'num_bus_nodes': _datum['bus'].x.shape[0],
            'num_ac_line_edges': _datum[('bus', 'ac_line', 'bus')].edge_index.shape[1],
            'num_transformer_edges': _datum[('bus', 'transformer', 'bus')].edge_index.shape[1],
            'dim_ac_line_edges': _datum[('bus', 'ac_line', 'bus')].edge_attr.shape[1],
            'dim_transformer_edges': _datum[('bus', 'transformer', 'bus')].edge_attr.shape[1],
            'out_channels': _datum['generator'].y.shape[-1],
        }
        opf_metadata = dataset[0].metadata()

        return train_loader, val_loader, test_loader, info, opf_metadata   