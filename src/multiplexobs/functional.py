import torch
import parametrization_cookbook.functions.torch as pcf


def log_norm_const(p, eps=1e-6):
    """
    Compute the logarithm of the normalization constant of a ContinuousBernoulli for a given input tensor.

    Args:
        p (torch.Tensor): Input tensor representing the probability values.
        eps (float, optional): Small value to avoid division by zero. Default is 1e-6.

    Returns:
        torch.Tensor: The logarithm of the normalization constant.

                    myMPO = estimation.pyramidal_training(data, nb_networks = L, nb_nodes = n,  obs_dist = 'Bernoulli', 
                                                        directed = False, is_dynamic = False,
                                                        nb_clusters = 5, 
                                                        nb_blocks_obs= 3, 
                                                        nb_blocks_lat = 3)    Raises:
        AssertionError: If any value in p is not between 0 and 1.
    """
    assert torch.all((0 <= p) & (p <= 1)), "p must be a probability value between 0 and 1"
    p = torch.clamp(p, eps, 1 - eps)
    p = torch.where((p < 0.49) | (p > 0.51), p, 0.49 *
        torch.ones_like(p))
    return torch.log((2 * torch.arctanh(1 - 2 * p)) /
            (1 - 2 * p) + eps)



def bernoulli_obs_likelihood(A_obs, M_obs, A_lat, M_lat, alpha_pos, alpha_neg, tau_nodes, tau_net):
    """
    Calculate the log-likelihood of observed data given the model parameters for a Bernoulli emission distribution (binary data).

    Args:
        A_obs (torch.Tensor): Observed adjacency matrix.
        M_obs (torch.Tensor): Mask matrix indicating which entries in A_obs are observed.
        A_lat (torch.Tensor): Latent adjacency matrix.
        M_lat (torch.Tensor): Mask matrix indicating which entries in A_lat are defined.
        alpha_pos (List[torch.Tensor]): List of parameters of the emission distribution (A_lat == 1) for each cluster.
        alpha_neg (List[torch.Tensor]): List of parameters of the emission distribution (A_lat == 0) for each cluster.
        tau_nodes (List[torch.Tensor]): List of node block memberships for each cluster.
        tau_net (torch.Tensor): Network clustering matrix.

    Returns:
        torch.Tensor: Log likelihood of the observed data given the model parameters.
    """
    ll = 0
    nb_clusters = len(tau_nodes)
    X1W = (M_obs * A_obs).permute(1,2,0).matmul(tau_net)
    X0W = (M_obs * (1 - A_obs)).permute(1,2,0).matmul(tau_net)
    for k in range(nb_clusters):            
        ZAP1Z = tau_nodes[k][:,:] @ alpha_pos[k][:,:].log() @ tau_nodes[k][:,:].t()
        ZAP0Z = tau_nodes[k][:,:] @ (1-alpha_pos[k][:,:]).log() @ tau_nodes[k][:,:].t()
        ZAN1Z = tau_nodes[k][:,:] @ alpha_neg[k][:,:].log() @ tau_nodes[k][:,:].t()
        ZAN0Z = tau_nodes[k][:,:] @ (1 - alpha_neg[k][:,:]).log() @ tau_nodes[k][:,:].t()
        ll += A_lat * X1W[:,:,k] * ZAP1Z
        ll += A_lat * X0W[:,:,k] * ZAP0Z
        ll += (1-A_lat) * X1W[:,:,k] * ZAN1Z
        ll += (1-A_lat) * X0W[:,:,k] * ZAN0Z
    ll = torch.sum(M_lat * ll)    
    return ll
    
def continuous_bernoulli_obs_likelihood(A_obs, M_obs, A_lat, M_lat, alpha_pos, alpha_neg, tau_nodes, tau_net):
    """
    Calculate the log-likelihood of observed data given the model parameters for a Continuous Bernoulli emission distribution (data in [0,1]).

    Args:
        A_obs (torch.Tensor): Observed adjacency matrix.
        M_obs (torch.Tensor): Mask matrix indicating which entries in A_obs are observed.
        A_lat (torch.Tensor): Latent adjacency matrix.
        M_lat (torch.Tensor): Mask matrix indicating which entries in A_lat are defined.
        alpha_pos (List[torch.Tensor]): List of parameters of the emission distribution (A_lat == 1) for each cluster.
        alpha_neg (List[torch.Tensor]): List of parameters of the emission distribution (A_lat == 0) for each cluster.
        tau_nodes (List[torch.Tensor]): List of node block memberships for each cluster.
        tau_net (torch.Tensor): Network clustering matrix.

    Returns:
        torch.Tensor: Log likelihood of the observed data given the model parameters.
    """
    ll = 0
    nb_clusters = len(tau_nodes)
    X1W = (M_obs * A_obs).permute(1,2,0).matmul(tau_net)
    X0W = (M_obs * (1 - A_obs)).permute(1,2,0).matmul(tau_net)
    for k in range(nb_clusters):            
        ZAP1Z = tau_nodes[k][:,:] @ alpha_pos[k][:,:].log() @ tau_nodes[k][:,:].t()
        ZAP0Z = tau_nodes[k][:,:] @ (1-alpha_pos[k][:,:]).log() @ tau_nodes[k][:,:].t()
        ZAN1Z = tau_nodes[k][:,:] @ alpha_neg[k][:,:].log() @ tau_nodes[k][:,:].t()
        ZAN0Z = tau_nodes[k][:,:] @ (1 - alpha_neg[k][:,:]).log() @ tau_nodes[k][:,:].t()
        ll += A_lat * X1W[:,:,k] * ZAP1Z
        ll += A_lat * X0W[:,:,k] * ZAP0Z
        ll += (1-A_lat) * X1W[:,:,k] * ZAN1Z
        ll += (1-A_lat) * X0W[:,:,k] * ZAN0Z
    XW = M_obs.permute(1,2,0).matmul(tau_net)
    for k in range(nb_clusters):
        ZAPZ = tau_nodes[k][:,:] @ log_norm_const(alpha_pos[k][:,:]) @ tau_nodes[k][:,:].t()
        ZANZ = tau_nodes[k][:,:] @ log_norm_const(alpha_neg[k][:,:]) @ tau_nodes[k][:,:].t()
        ll += A_lat * XW[:,:,k] * ZAPZ
        ll += (1 - A_lat) * XW[:,:,k] * ZANZ    
    ll = torch.sum(M_lat * ll)    
    return ll
    
    
    
def beta_obs_likelihood(A_obs, M_obs, A_lat, M_lat, alpha_pos, alpha_neg, tau_nodes, tau_net, beta_ss):
    """
    Calculate the log-likelihood of observed data given the model parameters for a Continuous Bernoulli emission distribution (data in [0,1]).

    Args:
        A_obs (torch.Tensor): Observed adjacency matrix.
        M_obs (torch.Tensor): Mask matrix indicating which entries in A_obs are observed.
        A_lat (torch.Tensor): Latent adjacency matrix.
        M_lat (torch.Tensor): Mask matrix indicating which entries in A_lat are defined.
        alpha_pos (List[torch.Tensor]): List of position parameters (A_lat == 1) for each cluster.
        alpha_neg (List[torch.Tensor]): List of position parameters (A_lat == 0) for each cluster.
        tau_nodes (List[torch.Tensor]): List of node block memberships for each cluster.
        tau_net (torch.Tensor): Network clustering matrix.
        beta_ss (torch.Tensor): Sample size parameter of the beta distribution.

    Returns:
        log-likelihood (torch.Tensor): Log likelihood of the observed data given the model parameters.
    """
    ll = 0
    nb_clusters = len(tau_nodes)
    beta_ss = 2 + pcf.softplus(beta_ss, scale = 1)
    X1W = (M_obs * torch.log(A_obs + 1e-9)).permute(1,2,0).matmul(tau_net)
    X0W = (M_obs * torch.log(1 - A_obs + 1e-9)).permute(1,2,0).matmul(tau_net)
    XW = M_obs.permute(1,2,0).matmul(tau_net)            
    for k in range(nb_clusters):
        ZAP1Z = tau_nodes[k][:,:] @ (alpha_pos[k][:,:] * beta_ss) @ tau_nodes[k][:,:].t()
        ZAP0Z = tau_nodes[k][:,:] @ ((1-alpha_pos[k][:,:]) * beta_ss) @ tau_nodes[k][:,:].t()
        ZAN1Z = tau_nodes[k][:,:] @ (alpha_neg[k][:,:] * beta_ss) @ tau_nodes[k][:,:].t()
        ZAN0Z = tau_nodes[k][:,:] @ ((1 - alpha_neg[k][:,:]) * beta_ss) @ tau_nodes[k][:,:].t()
        ll += A_lat * X1W[:,:,k] * ZAP1Z
        ll += A_lat * X0W[:,:,k] * ZAP0Z
        ll += (1-A_lat) * X1W[:,:,k] * ZAN1Z
        ll += (1-A_lat) * X0W[:,:,k] * ZAN0Z
        ZAPZ = tau_nodes[k][:,:] @ (torch.lgamma(beta_ss) - \
            torch.lgamma(beta_ss*alpha_pos[k][:,:]) - torch.lgamma(beta_ss * (1- alpha_pos[k][:,:]))) \
            @ tau_nodes[k][:,:].t()
        ZANZ = tau_nodes[k][:,:] @ (torch.lgamma(beta_ss) -  \
            torch.lgamma(beta_ss*alpha_neg[k][:,:]) - torch.lgamma(beta_ss * (1- alpha_neg[k][:,:]))) \
            @ tau_nodes[k][:,:].t()
        ll += A_lat * XW[:,:,k] * ZAPZ
        ll += (1 - A_lat) * XW[:,:,k] * ZANZ    
                
    ll = torch.sum(M_lat * ll)        

    return ll

def bernoulli_sbm_likelihood(A, M, alpha, tau):
    """
    Calculate the log-likelihood of a Bernoulli SBM network.

    Args:
        A (torch.Tensor): Adjacency matrix of the network.
        M (torch.Tensor): Mask matrix indicating which entries of A are observed.
        alpha (torch.Tensor): Parameters of the Bernoulli distribution.
        tau (torch.Tensor): Community assignment matrix.

    Returns:
        torch.Tensor: The likelihood of the Bernoulli SBM network.
    """

    ll = torch.sum(M * A * (tau @ alpha.logit() @ tau.t()))
    ll +=  torch.sum(M * (tau @ (1-alpha).log() @ tau.t()))

    return ll



def pen_inter(x, directed):
    if directed:
        return x*x
    else:
        return .5*x*(x+1) 