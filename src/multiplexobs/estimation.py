
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from multiplexobs.multiplexobs import MultiPlexObs
import numpy as np



def init_from_models(model1, model2, clamp_min=-1, clamp_max=1):
    """
    Initializes the parameters of model1 using the estimated parameters of model2.

    Args:
        model1 (MultiPlexObs): The target model to initialize.
        model2 (MultiPlexObs): The source model to extract parameters from.
        clamp_min (float, optional): The minimum value to clamp the initialized parameters. Defaults to -1.
        clamp_max (float, optional): The maximum value to clamp the initialized parameters. Defaults to 1.

    Returns:
        MultiPlexObs: The initialized model1.
    """
    
    model1.A_lat.add_(model2.A_lat.detach()).clamp_(1 / (1 + np.exp(clamp_max - clamp_min)),
                                                    1 / (1 + np.exp(clamp_min - clamp_max)))   
    with torch.no_grad():
        tau_lat = model2.tau_lat.detach()
        tau_net = model2.tau_net.detach()
        tau_obs = [model2.tau_obs[k].detach() for k in range(model2.nb_clusters)]
        L = model1.nb_networks
        n = model1.nb_nodes
        K1 = model1.nb_clusters
        K2 = model2.nb_clusters
        Q1 = model1.nb_blocks_obs[0]
        Q2 = model2.nb_blocks_obs[0]
        QA1 = model1.nb_blocks_lat
        QA2 = model2.nb_blocks_lat

        if QA1 > QA2:
            tau_lat = torch.cat( (tau_lat, torch.zeros((n, QA1 - QA2))), 1)
        if QA1 < QA2:
            tau_lat = tau_lat[:,: (QA1 - 1)]
        model1.tau_lat.add_(tau_lat).clamp_(clamp_min,clamp_max)


        if K1 > K2:
            tau_net = torch.cat((tau_net, torch.zeros((L, K1 - K2))), 1)
        if K1 < K2:
            tau_net = tau_net[:,: (K1 - 1)]
        model1.tau_net.add_(tau_net).clamp_(clamp_min,clamp_max)        

        if Q1 > Q2:
            tau_obs = [torch.cat((tau_obs[k], torch.zeros((n, Q1 - Q2))), 1) \
                for k in range(K2)]
        if Q1 < Q2:
            tau_obs = [tau_obs[k][:,: (model1.nb_blocks_lat - 1)] for k in range(K2)]
        
        
        if K1 > K2:
            [model1.tau_obs[k].add_(tau_obs[k]).clamp_(clamp_min,clamp_max) for k in range(K2)]    
        if K1 <= K2:
            [model1.tau_obs[k].add_(tau_obs[k]).clamp_(clamp_min,clamp_max) for k in range(K1)]                
    
    return model1



def pyramidal_training(data,
                       nb_networks,
                       nb_nodes,
                       nb_clusters = 1,
                       nb_blocks_obs = 1,
                       nb_blocks_lat = 1,
                       obs_dist = 'Bernoulli',
                       directed = False,
                       is_dynamic = False,
                       is_hierarchical =False,
                       net_covariates = None,
                       device = 'cpu',
                       early_stopping = True,
                       nb_init = 100, 
                       nb_run = 10, 
                       nb_epochs_init = 10,
                       nb_epochs_run = 1000,
                       batch_ratio_init = 5, 
                       batch_ratio_run = 1,
                       lr_init = 1,
                       lr_run = 0.05,
                       model2 = None,
                       **kwargs
                       ):
    """
    Trains a pyramidal network model using the given data.

    Args:
        data (DataLoader): The input data for training.
        nb_networks (int): The number of networks.
        nb_nodes (int): The number of nodes.
        nb_clusters (int): The number of clusters (default: 1).
        nb_blocks_obs ([int]): The number of blocks for observations (default: 1).
        nb_blocks_lat (int): The number of blocks for networks (default: 1).
        obs_dist (str): The distribution of observations (default: 'Bernoulli').
        directed (bool): Whether the network is directed (default: False).
        is_dynamic (bool): Whether the network is dynamic (default: False).
        is_hierarchical (bool): Whether the network is hierarchical (default: False).
        net_covariates ([int]): Whether the network is hierarchical (default: False).
        device (str): The device to use for training (default: 'cpu').
        early_stopping (bool): Whether to use early stopping during training (default: True).
        nb_init (int): The number of initializations (default: 100).
        nb_run (int): The number of runs (default: 10).
        nb_epochs_init (int): The number of epochs for initialization (default: 10).
        nb_epochs_run (int): The number of epochs for running (default: 1000).
        batch_ratio_init (int): The batch ratio for initialization (default: 5).
        batch_ratio_run (int): The batch ratio for running (default: 1).
        lr_init (double): The learning rate for initialization (default: 1).
        lr_run (double): The learning rate for running (default: 0.05).
        model2 (MultiPlexObs): The second model to initialize from (default: None).
        **kwargs: Additional arguments to be passed to the init_from_models function.

    Returns:
        MultiPlexObs: The fitted model with the lowest loss.
    """
    
    model_list = []
    icl_list = []
    tmp_model_list = []
    nb_blocks_obs = np.repeat(nb_blocks_obs, nb_clusters)
    if isinstance(nb_networks, torch.Tensor):
        nb_networks = int(nb_networks.numpy())
    for m in tqdm(range(nb_init), desc ='Initializing'):
        myMPO = MultiPlexObs(nb_networks=nb_networks, 
                            nb_nodes = nb_nodes, 
                            nb_clusters = nb_clusters,
                            nb_blocks_obs= nb_blocks_obs, 
                            nb_blocks_lat = nb_blocks_lat, 
                            obs_dist = obs_dist, 
                            directed = directed, 
                            is_hierarchical = is_hierarchical, 
                            is_dynamic = is_dynamic,
                            net_covariates=net_covariates,
                            device=device)                               
        myMPO.initialize(data)
        myMPO.to(device)
        if model2 is not None:
            init_from_models(myMPO, model2, **kwargs)
        optim = torch.optim.Adam(myMPO.params, lr = lr_init)
        myMPO.train(DataLoader(data, batch_size= int(nb_networks//batch_ratio_init), 
                            shuffle=True), 
                    optim, 
                    nb_epochs=nb_epochs_init, loss = 'elbo', verbose=True)
        tmp_model_list.append(myMPO)
#            bar.next()
    ord = np.array(np.argsort(np.array([tmp_model_list[m].loss_list[-1] for m in range(nb_init)])))[:nb_run]
    best_models = np.array(tmp_model_list)[ord]


    for m in tqdm(range(nb_run), desc ='Running'):
        myMPO = best_models[m]
        optim =torch.optim.Adam(myMPO.params, lr = lr_run) 
        myMPO.train(DataLoader(data, batch_size = int(nb_networks//batch_ratio_run), 
                            shuffle=True), 
            optim, 
            loss = 'elbo', 
            nb_epochs = nb_epochs_run, verbose = False, early_stopping=early_stopping)
        myMPO.train(DataLoader(data, batch_size = int(nb_networks), 
                            shuffle=True), 
            optim, 
            loss = 'elbo', 
            nb_epochs = nb_epochs_run, verbose = True, early_stopping=early_stopping)
        model_list.append(myMPO)
        icl_list.append(myMPO.icl())
    best_model_id = np.nanargmin(np.array([model_list[m].loss_list[-1] for m in range(nb_run)]))
    bestMPO = best_models[best_model_id]       
       
    return bestMPO


def train_models(data, min, max, step=1, dim="clusters", model2=None, depth=2, verbose=True, save=False, save_path = "save_folder", **kwargs):
    """
    Train multiple models using the MultiPlexObs class.

    Args:
        data (torch.Tensor): The input data for training the models.
        min (int): The minimum value for the range of the parameter to be varied.
        max (int): The maximum value for the range of the parameter to be varied.
        step (int, optional): The step size for the range of the parameter. Defaults to 1.
        dim (str, optional): The dimension to vary the parameter. Can be one of 'clusters', 'blocks_obs', or 'blocks_lat'. Defaults to 'clusters'.
        model2 (MultiPlexObs, optional): An object of class MultiPlexObs. Defaults to None.
        depth (int, optional): The depth of the stopping criterion. Defaults to 2.
        verbose (bool, optional): Whether to print the progress and results. Defaults to True.
        save (bool, optional): Whether to save the trained models. Defaults to False. Each model will be saved in a separate file.
        model_path (str, optional): The save path of the models. Defaults to "save_folder".
        **kwargs: Additional keyword arguments to be passed to the pyramidal_training function.

    Returns:
        list: A list of trained models, including the initial model.

    Raises:
        AssertionError: If model2 is not an object of class MultiPlexObs.
    """
    
    assert isinstance(model2, MultiPlexObs), "model2 must be an object of class MultiPlexObs"
    best_list = [model2]
    counter = 0
    for k in tqdm(range(min, max, step), desc='Train models'):
        if dim == 'clusters':
            K = k
            Q = model2.nb_blocks_obs[0]
            QA = model2.nb_blocks_lat
        elif dim == 'blocks_obs':
            K = model2.nb_clusters
            Q = k
            QA = model2.nb_blocks_lat        
        elif dim == 'blocks_lat':
            K = model2.nb_clusters
            QA = k
            Q = model2.nb_blocks_obs[0]
        bestMPO = pyramidal_training(data=data,
                                     nb_networks=model2.nb_networks,
                                     nb_nodes=model2.nb_nodes,
                                     nb_clusters=K,
                                     nb_blocks_obs=Q,
                                     nb_blocks_lat=QA,
                                     obs_dist=model2.obs_dist,
                                     directed=model2.directed,
                                     is_dynamic=model2.is_dynamic,
                                     is_hierarchical=model2.is_hierarchical,
                                     net_covariates=model2.net_covariates,
                                     model2=model2,
                                     **kwargs
                                     )   
        optim = torch.optim.Adam(bestMPO.params, lr=.05) 
        bestMPO.train(DataLoader(data, batch_size=int(model2.nb_networks.item()), shuffle=True), 
                      optim, 
                      loss='elbo', nb_epochs=5000, verbose=False)
        best_list.append(bestMPO)
        tmpMPO = bestMPO

        if verbose:
            print('[%d/%d/%d] icl: %.2f, loss: %.2f, entropy: %.2f, penalty: %.2f' %
                  (K, Q, QA,
                   bestMPO.complete_log_likelihood_list[-1] - bestMPO.penalty.numpy(),
                   bestMPO.loss_list[-1],
                   bestMPO.entropy_list[-1],
                   bestMPO.penalty.numpy()
                   ))
        if save:
            model_path = '{}_{}_{}_{}_{}_{}_{}.pt'.format(save_folder,
                                                           K, Q, QA, 
                                                           tmpMPO.is_hierarchical, 
                                                           tmpMPO.is_dynamic, tmpMPO.is_hierarchical)
            torch.save(tmpMPO, model_path)
        if best_list[-1].complete_log_likelihood_list[-1] - best_list[-1].penalty.numpy() > \
           best_list[-2].complete_log_likelihood_list[-1] - best_list[-2].penalty.numpy():
            counter = 0
        else:
            counter += 1
            if counter >= depth:
                break
    return best_list


