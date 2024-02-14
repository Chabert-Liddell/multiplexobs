import torch
import numpy as np
from multiplexobs.estimation import train_models
from multiplexobs.loader import MultiplexNetworkDataset
from multiplexobs.multiplexobs import MultiPlexObs
from torch.utils.data import DataLoader


L = 6
N = 20
device = 'cpu'
X = torch.randint(0, 2, (L, N, N), dtype=torch.float32)
X2 = torch.rand((L, N, N))
M = torch.ones_like(X)
M = M.tril(-1)
M[:,0:2] = 0
M[0:2,:] = 0
M2 = torch.ones_like(X2)
data = MultiplexNetworkDataset(X.to(device), M.to(device), np.ones(L), 
                                        directed=False, diagonal=False)
data2 = MultiplexNetworkDataset(X2.to(device), M.to(device), np.ones(L), 
                                        directed=True, diagonal=True)
myMPO = MultiPlexObs(nb_networks=L, 
                                nb_nodes = N, 
                                nb_clusters = 1, 
                                nb_blocks_obs= [1], 
                                nb_blocks_lat = 1, 
                                obs_dist = 'Bernoulli', 
                                directed = False, 
                                is_hierarchical = False, 
                                is_dynamic = False,
                                net_covariates=None,
                                device='cpu')
myMPO.initialize(data)
myMPO.to(device)
optim = torch.optim.Adam(myMPO.params, lr = 1)

myMPO.train(DataLoader(data, batch_size= L, shuffle=True), 
            optim, 
            nb_epochs=10, loss = 'elbo', verbose=False)               


myMPO2 = MultiPlexObs(nb_networks=L, 
                                nb_nodes = N, 
                                nb_clusters = 1, 
                                nb_blocks_obs= [1], 
                                nb_blocks_lat = 1, 
                                obs_dist = 'ContinuousBernoulli', 
                                directed = True, 
                                is_hierarchical = False, 
                                is_dynamic = False,
                                net_covariates=np.array([0,0,1,1,1]),
                                device='cpu')
myMPO2.initialize(data2)
myMPO2.to(device)            
optim = torch.optim.Adam(myMPO2.params, lr = 1)
myMPO2.train(DataLoader(data2, batch_size= L, shuffle=True), 
            optim, 
            nb_epochs=10, loss = 'elbo', verbose=False)    

myMPO3 = MultiPlexObs(nb_networks=L, 
                                nb_nodes = N, 
                                nb_clusters = 3, 
                                nb_blocks_obs= [1,1,1], 
                                nb_blocks_lat = 2, 
                                obs_dist = 'Beta', 
                                directed = True, 
                                is_hierarchical = True, 
                                is_dynamic = True,
                                net_covariates=None,
                                device='cpu')
myMPO3.initialize(data2)
myMPO3.to(device)            
optim = torch.optim.Adam(myMPO3.params, lr = 1)
myMPO3.train(DataLoader(data2, batch_size= L, shuffle=True), 
            optim, 
            nb_epochs=10, loss = 'elbo', verbose=False)    




def test_train_models():
    # Test case 1: Training models for clusters and missing nodes
    models = train_models(data, 2, 6, model2=myMPO, 
                          nb_init = 1, 
                       nb_run = 1, 
                       nb_epochs_init = 1,
                       nb_epochs_run = 1, batch_ratio_init = 2, lr_init = .1, depth = 5)
    assert np.allclose([1,2,3,4,5], [mod.nb_clusters for mod in models])  # Including the initial model

    # Test case 2: Training models with blocks_obs with different parameter ranges and missing nodes
    models = train_models(data, 2, 5, step=2, dim='blocks_obs', model2=myMPO,
                        nb_init = 1, 
                        nb_run = 1, 
                        nb_epochs_init = 1,
                        nb_epochs_run = 1, batch_ratio_init = 3, lr_init = .1, depth = 5)
    assert np.allclose([1,2,4], [mod.nb_blocks_obs[0] for mod in models]) # Including the initial model

    # Test case 3: Training models with blocks_lat with different parameter ranges and missing nodes
    models = train_models(data, 1, 4, step=1, dim='blocks_lat', model2=myMPO,
                        nb_init = 1, 
                        nb_run = 1, 
                        nb_epochs_init = 1,
                        nb_epochs_run = 1, batch_ratio_init = 2, lr_init=.1, depth = 5)
    assert np.allclose([1,1,2,3], [mod.nb_blocks_lat for mod in models]) # Including the initial model

    # Test case 4: Training models with ContinuousBernoulli distribution and net_covariates
    models = train_models(data, 2, 4, step=1, dim='clusters', model2=myMPO2,
                        nb_init = 1, 
                        nb_run = 1, 
                        nb_epochs_init = 1,
                        nb_epochs_run = 1, batch_ratio_run = 2,  batch_ratio_init = 2, lr_init = .1, depth = 5)
    assert np.allclose([(2,1),(2,2),(2,3)], [mod.natural_parameters_network().shape for mod in models]) # Including the initial model
    
    # Test case 4: Training models with beta distribution, dynamic and hierarchical
    models = train_models(data, 2, 4, step=1, dim='blocks_obs', model2=myMPO3,
                        nb_init = 1, 
                        nb_run = 1, 
                        nb_epochs_init = 1,
                        nb_epochs_run = 1, batch_ratio_run = 2,  batch_ratio_init = 2, lr_init = .1, depth = 5)
    assert np.allclose([(3,3),(3,3),(3,3)], [mod.natural_parameters_network().shape for mod in models]) # Including the initial model
    assert np.allclose([(2,1),(2,2),(2,3)], [mod.natural_parameters_obs()[0][0].shape for mod in models]) # Including the initial model
    
