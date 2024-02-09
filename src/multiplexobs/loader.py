import torch
from torch.utils.data import Dataset
import numpy as np

class MultiplexNetworkDataset(Dataset):
    ''' Dataset class; A data loader for iterating over the layers of a multiplex network.
    
    Args:
        adj_tensor (torch.Tensor): The adjacency tensor representing the multiplex network.
        mask_tensor (torch.Tensor, optional): The mask tensor indicating which entries in the adjacency tensor are valid. Defaults to None.
        labels (numpy.ndarray, optional): The labels associated with each network in the multiplex network. Defaults to None.
        directed (bool, optional): Whether the multiplex network is directed. Defaults to True.
        diagonal (bool, optional): Whether the diagonal entries in the adjacency tensor should be preserved. Defaults to False.
    '''
    def __init__(self, adj_tensor, mask_tensor=None, labels=None, directed=True, diagonal=False):
        assert adj_tensor.size()[2] == adj_tensor.size()[1]
        if mask_tensor is not None:
            assert adj_tensor.size() == mask_tensor.size()
        self.A = adj_tensor
        if mask_tensor is None:
            self.M = torch.ones_like(self.A)
        else:
            self.M = mask_tensor
        if labels is None:
            self.labels = 0 * np.arange(self.A.size()[0])
        else:
            self.labels = labels
        self.net_id = np.arange(self.A.size()[0])
        for m in np.arange(self.A.size()[0]):
            if directed:
                if diagonal:
                    self.A[m,:,:] = self.A[m,:,:]
                    self.M[m,:,:] = self.M[m,:,:]
                else:
                    self.A[m,:,:].fill_diagonal_(0)
                    self.M[m,:,:].fill_diagonal_(0)                             
            elif diagonal:
                self.A[m,:,:].tril_(diagonal=0)
                self.M[m,:,:].tril_(diagonal=0)
            else:            
                self.A[m,:,:].tril_(diagonal=-1)
                self.M[m,:,:].tril_(diagonal=-1)        
    
    def __len__(self):
        return len(self.net_id)
    
    def __getitem__(self, idx):
        Adj = self.A[idx,:,:]
        Mask = self.M[idx,:,:]
        net_id = self.net_id[idx]
        labels = self.labels[idx]
        return Adj, Mask, net_id, labels
