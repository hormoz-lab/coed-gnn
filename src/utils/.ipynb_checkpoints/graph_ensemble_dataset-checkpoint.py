from torch_geometric.data import Data, InMemoryDataset


class GraphEnsembleDataset(InMemoryDataset):
    
    def __init__(self, 
        root, x, y, edge_index, 
        mask=None, edge_attr=None, num_nodes=None, transform=None,
    ):
        self.x = x
        self.y = y
        self.edge_index = edge_index
        self.transform = transform
        self.mask = mask 
        self.edge_attr = edge_attr 
        self.num_nodes = num_nodes
        super().__init__(root, transform, None)
        self.data, self.slices = self.collate([self.get(idx) for idx in range(len(self))])

    def len(self):
        return len(self.x)
        

    def get(self, idx):
        data = Data(
            x=self.x[idx],  
            edge_index=self.edge_index,  
            y=self.y[idx],  
        )
    
        if self.mask is not None:
            data.mask = self.mask[idx]
    
        if self.edge_attr is not None:
            data.edge_attr = self.edge_attr[idx]

        if self.transform is not None:
            data = self.transform(data)
    
        return data