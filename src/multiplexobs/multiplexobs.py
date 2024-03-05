import torch
import torch.nn as nn
import parametrization_cookbook.functions.torch as pcf
import numpy as np
import multiplexobs.functional as F


class MultiPlexObs(nn.Module):
    def __init__(
        self,
        nb_networks,
        nb_nodes,
        nb_clusters,
        nb_blocks_obs,
        nb_blocks_lat,
        obs_dist,
        directed,
        is_hierarchical=False,
        is_dynamic=False,
        net_covariates=None,
        device="cpu",
    ):
        """Initialize the MultiplexObs model.

        Args:
            nb_networks (int): The number of networks.
            nb_nodes (int): The number of nodes of each network.
            nb_clusters (int): The number of clusters of networks.
            nb_blocks_obs ([int]): A list with the number of blocks of each observation process.
            nb_blocks_lat (int): The number of blocks of the true network.
            obs_dist (str): Either "Bernoulli" for binary data or "ContinuousBernoulli" for data in (0,1).
            directed (bool): Are the networks symmetric, i.e. data[t,i,j] == data[t,j,i] for all i, j, t?
            is_hierarchical (bool, optional): Is there a hierarchical relationship between the blocks of the
                    true network and the blocks of the observations processes. Defaults to False.
            is_dynamic (bool, optional): If true, then the ordering of the networks matters. There is a Markovian
                    dynamic in the observation process: the cluster of a network depends on the cluster of the previous
                    network. Network 0 is in cluster 0. Defaults to False.
            net_covariates (None or array-like, optional): Covariates for the networks. Defaults to None.
            device (str, optional): Device of the torch object. Defaults to 'cpu'.
        """
        super().__init__()
        
        assert (
            len(nb_blocks_obs) == nb_clusters
        ), "The length of nb_blocks_obs must be the same as nb_clusters!"
        assert np.all(
            [q == nb_blocks_obs[0] for q in nb_blocks_obs]
        ), "Each observation process must have the same number of blocks!"
        assert (not (net_covariates is not None) &  is_dynamic
        ), "Model can not have both net_covariates and is_dynamic. Please choose one."
        if net_covariates is not None:
            assert (len(net_covariates) == nb_networks), "Length of net_covariates must be the same as nb_networks!"
        if isinstance(nb_nodes, torch.Tensor):
            self.nb_nodes = nb_nodes 
        else: 
            self.nb_nodes = torch.tensor(nb_nodes)
        if isinstance(nb_networks, torch.Tensor):
            self.nb_networks = nb_networks 
        else: 
            self.nb_networks = torch.tensor(nb_networks)        
        if isinstance(nb_clusters, torch.Tensor):
            self.nb_clusters = nb_clusters 
        else: 
            self.nb_clusters = torch.tensor(nb_clusters)
        if isinstance(nb_blocks_obs, torch.Tensor):
            self.nb_blocks_obs = nb_blocks_obs 
        else: 
            self.nb_blocks_obs = torch.tensor(nb_blocks_obs)
        if isinstance(nb_blocks_lat, torch.Tensor):
            self.nb_blocks_lat = nb_blocks_lat 
        else: 
            self.nb_blocks_lat = torch.tensor(nb_blocks_lat)
        self.is_hierarchical = is_hierarchical
        self.is_dynamic = is_dynamic
        self.obs_dist = obs_dist
        self.directed = directed
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cpu")

        self.net_covariates = net_covariates
        if self.net_covariates is not None:
            self.nb_covariates = np.max(net_covariates) + 1

        if self.obs_dist == "Beta":
            self.beta_ss = (
                torch.randn(1, dtype=torch.float32).to(self.device).requires_grad_(True)
            )
        self.tau_lat = (
            torch.randn(nb_nodes, nb_blocks_lat - 1, dtype=torch.float32)
            .to(self.device)
            .requires_grad_(True)
        )
        self.tau_obs = [
            torch.randn(nb_nodes, q - 1, dtype=torch.float32)
            .to(self.device)
            .requires_grad_(True)
            for q in nb_blocks_obs
        ]
        self.tau_net = (
            torch.randn(nb_networks, nb_clusters - 1, dtype=torch.float32)
            .to(self.device)
            .requires_grad_(True)
        )
        self.alpha_obs_pos = [
            (1 + torch.randn(q, q, dtype=torch.float32))
            .to(self.device)
            .requires_grad_(True)
            for q in nb_blocks_obs
        ]
        self.alpha_obs_neg = [
            (-1 + torch.randn(q, q, dtype=torch.float32))
            .to(self.device)
            .requires_grad_(True)
            for q in nb_blocks_obs
        ]
        self.alpha_lat = (
            torch.randn(nb_blocks_lat, nb_blocks_lat, dtype=torch.float32)
            .to(self.device)
            .requires_grad_(True)
        )
        self.pi_lat = (
            torch.randn(nb_blocks_lat - 1, dtype=torch.float32)
            .to(self.device)
            .requires_grad_(True)
        )

        if self.is_hierarchical:
            self.pi_obs = [
                torch.randn(nb_blocks_lat, q - 1, dtype=torch.float32)
                .to(self.device)
                .requires_grad_(True)
                for q in nb_blocks_obs
            ]
        else:
            self.pi_obs = [
                torch.randn(1, q - 1, dtype=torch.float32)
                .to(self.device)
                .requires_grad_(True)
                for q in nb_blocks_obs
            ]

        if self.is_dynamic:
            self.pi_net = (
                torch.randn(nb_clusters, nb_clusters - 1, dtype=torch.float32)
                .to(self.device)
                .requires_grad_(True)
            )
        elif net_covariates is not None:
            self.pi_net = (
                torch.randn(self.nb_covariates, nb_clusters - 1, dtype=torch.float32)
                .to(self.device)
                .requires_grad_(True)
            )
        else:
            self.pi_net = (
                torch.randn(nb_clusters - 1, dtype=torch.float32)
                .to(self.device)
                .requires_grad_(True)
            )

        self.params = [
            self.tau_lat,
            *self.tau_obs,
            self.tau_net,
            *self.alpha_obs_pos,
            *self.alpha_obs_neg,
            self.alpha_lat,
            self.pi_lat,
            *self.pi_obs,
            self.pi_net,
        ]
        if self.obs_dist == "Beta":
            self.params.append(self.beta_ss)

    def initialize(self, data):
        """
        Initializes the multiplex observation model with the given data.

        Args:
            data: The input data containing the adjacency tensors (A) and the mask tensor (M).

        Returns:
        None
        """
        self.M_lat = data.M.max(dim=0)[0].to(self.device)
        self.nb_inter_lat = self.M_lat.sum()
        self.nb_inter_obs = data.M.sum()
        self.nb_nodes_obs = (1 * (data.M.sum(dim=1) + data.M.sum(dim=2)) > 0).sum()
        self.present = 1.0 * (data.M.sum(dim=1) + data.M.sum(dim=2) > 0)
        self.p_present = [
            self.present.mean() + torch.zeros(q, dtype=torch.float32).to(self.device)
            for q in self.nb_blocks_obs
        ]
        self.nb_labels = np.unique(data.labels)
        self.A_lat = (data.M * data.A).sum(dim=0) / (data.M.sum(dim=0) + 1)
        self.penalty = self.compute_penalty()
        self.missing_nodes = self.nb_nodes_obs < self.nb_networks*self.nb_nodes


    def initialize_variational(
        self, tau_obs=None, tau_net=None, tau_lat=None, init_type="random"
    ):
        """Initialize the variational parameters. Useful to import variational parameters estimated from another model on the same data.

        Args:
            tau_obs (list, optional): List of observation variational parameters. Defaults to None.
            tau_net (float, optional): Network clustering variational parameters. Defaults to None.
            tau_lat (float, optional): Latent network variational parameters. Defaults to None.
            init_type (str, optional): Initialization type. 'random' adds a normal gaussian noise to each parameter. Defaults to 'random'.
        """
        with torch.no_grad():
            if tau_net is not None:
                tau_net = pcf.simplex_to_reals(tau_net)
                if init_type == "random":
                    tau_net.clamp_(-2, 2)
                    self.tau_net.add_(tau_net)
                else:
                    tau_net.clamp_(-10, 10)
                    self.tau_net.mul_(0).add_(tau_net)
            if tau_lat is not None:
                tau_lat = pcf.simplex_to_reals(tau_lat)
                if init_type == "random":
                    tau_lat.clamp_(-2, 2)
                    self.tau_lat.add_(tau_lat)
                else:
                    tau_lat.clamp_(-10, 10)
                    self.tau_lat.mul_(0).add_(tau_lat)
            if tau_obs is not None:
                tau_obs = [
                    pcf.simplex_to_reals(tau_obs[k]) for k in range(self.nb_clusters)
                ]
                if init_type == "random":
                    [tau_obs[k].clamp_(-2, 2) for k in range(self.nb_clusters)]
                    [self.tau_obs[k].add_(tau_obs[k]) for k in range(self.nb_clusters)]
                else:
                    [tau_obs[k].clamp_(-10, 10) for k in range(self.nb_clusters)]
                    [
                        self.tau_obs[k].mul_(0).add_(tau_obs[k])
                        for k in range(self.nb_clusters)
                    ]

    def natural_parameters_obs(self):
        """
        Computes the natural parameters for the observed networks.
        Simplex are encoded into probability vectors.

        Returns:
            pi_obs (List[torch.Tensor]): List of mixture parameters for each cluster.
            alpha_obs_pos (List[torch.Tensor]): List of emission parameters given a latent interaction (sigmoid transformed parameters) for each cluster.
            alpha_obs_neg (List[torch.Tensor]): List of emission parameters given no latent interaction (sigmoid transformed parameters) for each cluster.
        """

        alpha_obs_pos = [
            torch.sigmoid(self.alpha_obs_pos[k]) for k in range(self.nb_clusters)
        ]
        alpha_obs_neg = [
            torch.sigmoid(self.alpha_obs_neg[k]) for k in range(self.nb_clusters)
        ]
        if self.directed == False:
            for k in range(self.nb_clusters):
                alpha_obs_pos[k] = (
                    alpha_obs_pos[k].tril() + alpha_obs_pos[k].tril(diagonal=-1).t()
                )
                alpha_obs_neg[k] = (
                    alpha_obs_neg[k].tril() + alpha_obs_neg[k].tril(diagonal=-1).t()
                )
        pi_obs = [pcf.reals_to_simplex(self.pi_obs[k]) for k in range(self.nb_clusters)]
        return pi_obs, alpha_obs_pos, alpha_obs_neg

    def natural_parameters_latent(self):
        """
        Compute the natural parameters for the latent network.

        Returns:
            pi_lat (torch.Tensor): The mixture parameters for the latent network.
            alpha_lat (torch.Tensor): The emission (prior probability) parameters for the latent network.
        """
        alpha_lat = torch.sigmoid(self.alpha_lat)
        if self.directed == False:
            alpha_lat = alpha_lat.tril() + alpha_lat.tril(diagonal=-1).t()
        pi_lat = pcf.reals_to_simplex(self.pi_lat)
        return pi_lat, alpha_lat

    def natural_parameters_network(self):
        """
        Converts the real-valued parameters of the network to their corresponding simplex representation.

        Returns:
            pi_net (torch.Tensor): The probability vector of network (observation) clustering.
        """
        pi_net = pcf.reals_to_simplex(self.pi_net)
        return pi_net

    def natural_parameters_entropy(self):
        """
        Converts the real-valued parameters into the natural parameterization of the variational parameters.

        Returns:
            tau_obs (List[torch.Tensor]): List of the variational parameters for the block memberships of shape (nb_nodes, nb_blocks_obs) for each cluster. Each
            row is a probability vector.
            tau_net (torch.Tensor): Variational parameters for the network clustering of shape (nb_networks, nb_clusters). Each
            row is a probability vector.
            tau_lat (torch.Tensor): Variational parameters for the latent network block memberships of shape (nb_nodes, nb_blocks_lat). Each
            row is a probability vector
        """
        tau_net = pcf.reals_to_simplex(self.tau_net)
        tau_lat = pcf.reals_to_simplex(self.tau_lat)
        tau_obs = [
            pcf.reals_to_simplex(self.tau_obs[k]) for k in range(self.nb_clusters)
        ]
        return tau_obs, tau_net, tau_lat

    def train(
        self,
        dataloader,
        optimizer,
        nb_epochs=500,
        loss="elbo",
        verbose=True,
        early_stopping=True
    ):
        """Train the model with a SGD algorithm with minibatch

        Args:
            dataloader (DataLoader): The data tensor need to be transform first into
                a MultiplexNetworkDataset object before passing it in a DataLoader object.
            optimizer (optimizer): A pytorch optimizer. Defaults to "torch.optim.Adam".
            nb_epochs (int, optional): The number of training epochs. Default to 500.
            loss (str, optional): Name of the function to optimize. Defaults to 'elbo'.
            verbose (bool, optional): Print options. Defaults to True.
        """

        assert loss == "elbo", "Please set loss to 'elbo'"

        self.loss = loss
        if self.loss == "ilvb":
            self.prior0 = torch.tensor(0.5)
            self.prior1 = torch.tensor(0.5)
        self.optimizer = optimizer
        if not hasattr(self, "loss_list"):
            self.loss_list = []
            self.entropy_list = []
            self.complete_log_likelihood_list = []
        self.epochs = nb_epochs

        for epoch in range(self.epochs):
            loss_tmp = 0
            complete_log_likelihood_tmp = 0
            entropy_tmp = 0
            for net, mask, id, labels in dataloader:
                data_ratio = len(id) / self.nb_networks
                if self.loss == "elbo":
                    entropy = self.entropy(id, labels)
                    complete_log_likelihood = self.complete_log_likelihood(
                        net, mask, id, labels
                    )
                    elbo = entropy + complete_log_likelihood
                    loss_ = -elbo

                optimizer.zero_grad()
 #               with torch.autograd.detect_anomaly():
                loss_.backward()
                optimizer.step()

                with torch.no_grad():
                    self.alpha_lat.clamp_(-5, 5)
                    for k in range(self.nb_clusters):
                        self.pi_obs[k].clamp_(-5, 5)
                        self.tau_obs[k].clamp_(-5, 5)
                        self.alpha_obs_pos[k].clamp_(-5, 5)
                        self.alpha_obs_neg[k].clamp_(-5, 5)
                    self.pi_lat.clamp_(-5, 5)
                    self.pi_net.clamp_(-5, 5)
                    self.tau_lat.clamp_(-5, 5)
                    self.tau_net.clamp_(-5, 5)
                    _, alpha_lat = self.natural_parameters_latent()
                    if alpha_lat.isnan().any():
                        raise RuntimeError("nan value while computing alpha_lat in natural_parameters_latent(). Try to decrease Adam learning_rate.")
                    _, alpha_obs_pos, alpha_obs_neg = self.natural_parameters_obs()
                    tau_obs, tau_net, tau_lat = self.natural_parameters_entropy()
                    for k in range(self.nb_clusters):
                        if alpha_obs_pos[k].isnan().any():
                            raise RuntimeError("nan value while computing alpha_lat in natural_parameters_latent(). Try to decrease Adam learning_rate.")
                        if alpha_obs_neg[k].isnan().any():
                            raise RuntimeError("nan value while computing alpha_lat in natural_parameters_latent(). Try to decrease Adam learning_rate.")
                        if tau_obs[k].isnan().any():
                            raise RuntimeError("nan value while computing tau_obs in natural_parameters_entropy(). Try to decrease Adam learning_rate.")
                    if tau_net.isnan().any():
                        raise RuntimeError("nan value while computing tau_net in natural_parameters_entropy(). Try to decrease Adam learning_rate.")
                    if tau_lat.isnan().any():
                        raise RuntimeError("nan value while computing tau_lat in natural_parameters_entropy(). Try to decrease Adam learning_rate.")
                    self.A_lat = (
                        1 - data_ratio
                    ) * self.A_lat + data_ratio * self.ve_step_A(
                        net,
                        mask,
                        tau_net[id, :],
                        tau_obs,
                        tau_lat,
                        alpha_obs_pos,
                        alpha_obs_neg,
                        alpha_lat,
                    )
                    if self.missing_nodes:
                        for k in range(self.nb_clusters):
                            self.p_present[k] = (1 - data_ratio) * self.p_present[
                                k
                            ] + data_ratio * (
                                tau_net[id, k] @ self.present[id, :] @ tau_obs[k]
                            ) / (
                                tau_net[id, k].sum(dim=0) * tau_obs[k].sum(dim=0)
                            )
                loss_tmp += loss_.item()
                entropy_tmp += entropy.item()
                complete_log_likelihood_tmp += complete_log_likelihood.item()

            with torch.no_grad():
                _, alpha_lat = self.natural_parameters_latent()
                _, alpha_obs_pos, alpha_obs_neg = self.natural_parameters_obs()
                tau_obs, tau_net, tau_lat = self.natural_parameters_entropy()
                if self.missing_nodes:
                    for k in range(self.nb_clusters):
                        self.p_present[k] = (
                            tau_net[:, k] @ self.present[:, :] @ tau_obs[k]
                        ) / (tau_net[:, k].sum(dim=0) * tau_obs[k].sum(dim=0))

            self.loss_list.append(loss_tmp)
            self.entropy_list.append(entropy_tmp)
            self.complete_log_likelihood_list.append(complete_log_likelihood_tmp)

            if verbose & (epoch % 10 == 9):
                print(
                    "[%d] loss: %.2f, loglik: %.2f, entropy: %.2f"
                    % (
                        epoch + 1,
                        np.mean(self.loss_list[-10:]),
                        np.mean(self.complete_log_likelihood_list[-10:]),
                        np.mean(self.entropy_list[-10:]),
                    )
                )

            if early_stopping:
                if epoch > 20:
                    if self.loss_list[-11] < np.min(self.loss_list[-10:]) + 0.01:
                        break

    def complete_log_likelihood(self, A, M, net_id, net_labels):
        """
        Calculates the complete log likelihood of the model.

        Args:
            A (torch.Tensor): Adjacency matrix of the observed network.
            M (torch.Tensor): Mask matrix indicating the presence of edges in the observed network.
            net_id (torch.Tensor): Network ID for each observation.
            net_labels (torch.Tensor): Network labels for each observation.

        Returns:
            torch.Tensor: The complete log likelihood value.
        """
        nb_net = len(net_id)
        pi_lat, alpha_lat = self.natural_parameters_latent()
        pi_net = self.natural_parameters_network()
        pi_obs, alpha_obs_pos, alpha_obs_neg = self.natural_parameters_obs()
        tau_obs, tau_net, tau_lat = self.natural_parameters_entropy()
        lat_net = self.A_lat

        cll = 0
        # obs likelihood
        cll += self.obs_likelihood(
            A,
            M,
            lat_net,
            self.M_lat,
            alpha_obs_pos,
            alpha_obs_neg,
            tau_obs,
            tau_net[net_id, :]
        )
        if cll.isnan():
            raise RuntimeError("nan value after computing obs_likelihood.")
        # Latent network likelihood
        cll += (nb_net / self.nb_networks) * self.net_likelihood(
            lat_net, self.M_lat, alpha_lat, tau_lat
        )
        if cll.isnan():
            raise RuntimeError("nan value after computing net_likelihood.")
        # Latent network mixture likelihood
        cll += (nb_net / self.nb_networks) * torch.sum(tau_lat @ pi_lat.log())
        if cll.isnan():
            raise RuntimeError("nan value after computing latent mixture.")
        # Observation block likelihood
        for k in range(self.nb_clusters):
            if self.is_hierarchical:
                cll += (nb_net / self.nb_networks) * torch.sum(
                    (tau_lat @ pi_obs[k].log()) * tau_obs[k]
                )
            else:
                cll += (nb_net / self.nb_networks) * torch.sum(
                    tau_obs[k] @ pi_obs[k][0, :].log()
                )
        if cll.isnan():
            raise RuntimeError("nan value after computing obs mixture.")
        if self.net_covariates is not None:
            cll += (nb_net / self.nb_networks) * torch.sum(
                torch.einsum(
                    "ij,ij->i", tau_net, pi_net.log()[self.net_covariates, :]
                )
                )
        else:
            if self.is_dynamic:
                cll += (nb_net / self.nb_networks) * (
                    torch.sum((tau_net[0:-1,].T @ tau_net[1:,]) * pi_net.log())
                )
            else:
                cll += torch.sum(tau_net[net_id, :] @ pi_net.log())

        if cll.isnan():
            raise RuntimeError("nan value after computing network mixture.")

        # Value for nodes class sampling
        if self.missing_nodes:
            for k in range(self.nb_clusters):
                cll += (
                    tau_net[net_id, k]
                    @ self.present[net_id, :]
                    @ tau_obs[k]
                    @ (self.p_present[k]+1e-12).log()
                )
                cll += (
                    tau_net[net_id, k]
                    @ (1 - self.present[net_id, :])
                    @ tau_obs[k]
                    @ (1 - self.p_present[k]+1e-12).log()
                )
        if cll.isnan():
            raise RuntimeError("nan value after computing missing nodes likelihood.")
        return cll

    def entropy(self, net_id, net_labels):
        nb_net = len(net_id)
        tau_obs, tau_net, tau_lat = self.natural_parameters_entropy()
        lat_net = self.A_lat

        entropy = 0
        for k in range(self.nb_clusters):
            entropy -= (nb_net / self.nb_networks) * torch.sum(
                torch.xlogy(tau_obs[k], tau_obs[k])
            )
        if self.is_dynamic:
            entropy -= (nb_net / self.nb_networks) * torch.sum(
                torch.xlogy(tau_net, tau_net)
            )
        else:
            entropy -= torch.sum(torch.xlogy(tau_net[net_id, :], tau_net[net_id, :]))
        entropy -= (nb_net / self.nb_networks) * torch.sum(
            torch.xlogy(tau_lat, tau_lat)
        )
        entropy -= (nb_net / self.nb_networks) * torch.sum(
            torch.xlogy(self.M_lat * lat_net, lat_net)
        )
        entropy -= (nb_net / self.nb_networks) * torch.sum(
            torch.xlogy(self.M_lat * (1 - lat_net), 1 - lat_net)
        )
        if entropy.isnan():
            raise RuntimeError("Nan value while computing entropy")
        return entropy

    def net_likelihood(self, A, M, alpha, tau):

        ll = F.bernoulli_sbm_likelihood(A, M, alpha, tau)
        if ll.isnan():
            raise RuntimeError("Nan value while computing net_likelihood")
        return ll

    def obs_likelihood(
        self, A_obs, M_obs, A_lat, M_lat, alpha_pos, alpha_neg, tau_nodes, tau_net
    ):

        if self.obs_dist == "Bernoulli":
            ll = F.bernoulli_obs_likelihood(
                A_obs, M_obs, A_lat, M_lat, alpha_pos, alpha_neg, tau_nodes, tau_net
            )
        if self.obs_dist == "ContinuousBernoulli":
            ll = F.continuous_bernoulli_obs_likelihood(
                A_obs, M_obs, A_lat, M_lat, alpha_pos, alpha_neg, tau_nodes, tau_net
            )
        if self.obs_dist == "Beta":
            ll = F.beta_obs_likelihood(
                A_obs,
                M_obs,
                A_lat,
                M_lat,
                alpha_pos,
                alpha_neg,
                tau_nodes,
                tau_net,
                self.beta_ss
            )
        if ll.isnan():
            raise RuntimeError("Nan value while computing obs_likelihood")
        return ll

    def ve_step_A(
        self, net, mask, q_net, q_obs, q_lat, alpha_pos, alpha_neg, alpha_lat
    ):
        """
        Performs the variational expectation (VE) step for updating the latent network A.

        Args:
            net (torch.Tensor): The network tensor.
            mask (torch.Tensor): The mask tensor.
            q_net (torch.Tensor): The variational distribution of the network tensor.
            q_obs (List[torch.Tensor]): The variational distributions of the observation tensors.
            q_lat (torch.Tensor): The variational distribution of the latent variable tensor.
            alpha_pos (List[torch.Tensor]): The positive parameters for the observation distributions.
            alpha_neg (List[torch.Tensor]): The negative parameters for the observation distributions.
            alpha_lat (torch.Tensor): The parameters for the latent variable distribution.

        Returns:
            torch.Tensor: The updated variational distribution of the latent variable A.
        """
        eps = 1e-12
        A_prior = q_lat @ alpha_lat.logit(eps = eps) @ q_lat.T
        f_obs_pos = 0
        f_obs_neg = 0
        L = q_net.shape[0]
        if self.obs_dist in ("Bernoulli", "ContinuousBernoulli"):
            for k in range(self.nb_clusters):
                X_pos = (q_net[:, k].view(L, 1, 1) * net * mask).sum(dim=0)
                X_neg = (q_net[:, k].view(L, 1, 1) * (1 - net) * mask).sum(dim=0)
                f_obs_pos += X_pos * (q_obs[k] @ (alpha_pos[k].clamp(eps,1-eps)).log() @ q_obs[k].T)
                f_obs_pos += X_neg * (q_obs[k] @ (1 - alpha_pos[k].clamp(eps,1-eps)).log() @ q_obs[k].T)
                f_obs_neg += X_pos * (q_obs[k] @ (alpha_neg[k].clamp(eps,1-eps)).log() @ q_obs[k].T)
                f_obs_neg += X_neg * (q_obs[k] @ (1 - alpha_neg[k].clamp(eps,1-eps)).log() @ q_obs[k].T)
                if self.obs_dist == "ContinuousBernoulli":
                    X = torch.sum(q_net[:, k].view(L, 1, 1) * mask, axis=0)
                    f_obs_pos += X * (
                        q_obs[k] @ F.log_norm_const(alpha_pos[k].clamp(eps,1-eps)) @ q_obs[k].T
                    )
                    f_obs_neg += X * (
                        q_obs[k] @ F.log_norm_const(alpha_neg[k].clamp(eps,1-eps)) @ q_obs[k].T
                    )
        if self.obs_dist == 'Beta':
            beta_ss = 2 + pcf.softplus(self.beta_ss, scale = 1)
            for k in range(self.nb_clusters):
                X_pos = (q_net[:, k].view(L, 1, 1) * net.clamp(eps,1-eps).log() * mask).sum(dim=0)
                X_neg = (q_net[:, k].view(L, 1, 1) * (1 - net.clamp(eps,1-eps)).log() * mask).sum(dim=0)
                f_obs_pos += X_pos * (q_obs[k] @ (alpha_pos[k].clamp(eps,1-eps)*beta_ss) @ q_obs[k].T)
                f_obs_pos += X_neg * (q_obs[k] @ (((1-alpha_pos[k].clamp(eps,1-eps))*beta_ss)) @ q_obs[k].T)
                f_obs_pos += q_obs[k] @ (torch.lgamma(beta_ss) - \
                    torch.lgamma(beta_ss*alpha_pos[k].clamp(eps,1-eps)) - \
                    torch.lgamma(beta_ss * (1- alpha_pos[k].clamp(eps,1-eps)))) \
                        @ q_obs[k].T
                f_obs_neg += X_pos * (q_obs[k] @ (alpha_neg[k].clamp(eps,1-eps) * beta_ss) @ q_obs[k].T)
                f_obs_neg += X_neg * (q_obs[k] @ ((1 - alpha_neg[k].clamp(eps,1-eps)) * beta_ss) @ q_obs[k].T)
                f_obs_neg += q_obs[k] @ (torch.lgamma(beta_ss) - \
                    torch.lgamma(beta_ss*alpha_neg[k].clamp(eps,1-eps)) - \
                    torch.lgamma(beta_ss * (1- alpha_neg[k].clamp(eps,1-eps)))) \
                        @ q_obs[k].T                            
#        Pos = -1 + A_prior + f_obs_pos - f_obs_neg
#        Neg = -1 - A_prior - f_obs_pos + f_obs_neg
#        Max = torch.fmax(Pos, Neg)
        q_A = torch.sigmoid(A_prior + f_obs_pos - f_obs_neg)
#        q_A = (Pos - Max).exp() / ((Pos - Max).exp() + (Neg - Max).exp())
#        q_A = torch.softmax(torch.stack((Pos + 1, - Pos - 1 )), dim=0)[0]        
        if q_A.isnan().any():
            raise RuntimeError("nan while computing A_lat")
        
        return q_A

    def compute_penalty(self):
        """Compute the ICL penalty of the model"""
        pen_obs_mix = 0
        pen_obs_con = 0
        pen_obs_mis = 0
        for k in range(self.nb_clusters):
            pen_obs_con += (
                0.5
                * F.pen_inter(self.nb_blocks_obs[k], self.directed)
                * self.nb_inter_obs.log()
            )
            if self.is_hierarchical:
                pen_obs_mix += (
                    0.5
                    * self.nb_blocks_lat
                    * (self.nb_blocks_obs[k] - 1)
                    * self.nb_nodes_obs.log()
                )
            else:
                pen_obs_mix += (
                    0.5 * (self.nb_blocks_obs[k] - 1) * self.nb_nodes_obs.log()
                )
            pen_obs_mis += 0.5 * self.nb_blocks_obs[k] * self.nb_nodes_obs.log()
        pen_lat_con = (
            0.5
            * F.pen_inter(self.nb_blocks_lat, self.directed)
            * self.nb_inter_lat.log()
        )
        pen_lat_mix = 0.5 * (self.nb_blocks_lat - 1) * self.nb_nodes.log()
        pen_net_mix = 0
        if self.is_dynamic:
            pen_net_mix += (
                0.5
                * self.nb_clusters
                * (self.nb_clusters - 1)
                * (self.nb_networks - 1).log()
            )
        else:
            if self.net_covariates is None:
                pen_net_mix += 0.5 * (self.nb_clusters - 1) * self.nb_networks.log()
            else:
                for c in range(self.nb_covariates):
                    pen_net_mix += (
                        0.5
                        * (self.nb_clusters - 1)
                        * np.log(np.sum(self.net_covariates == c))
                    )

        # Store penalty
        penalty_obs = pen_obs_mis + pen_obs_con + pen_obs_mis
        penalty_net = pen_net_mix
        penalty_lat = pen_lat_mix + pen_lat_con
        penalty = penalty_lat + penalty_net + penalty_obs
        return penalty

    def icl(self):
        icl = self.complete_log_likelihood_list[-1] - self.penalty
        return icl.item()
