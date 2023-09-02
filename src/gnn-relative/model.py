import numpy as np
from scipy.sparse import coo_matrix
import torch
from torch import nn
from torch_geometric.nn import MetaLayer
from torch_scatter import scatter_mean
from tqdm import tqdm
from utils import makeGraphfromTraj, plotGraph, doUpdate, prepareEForModel, reverseNormalizeX, normalizeX
from data import DatasetGraph, DataloaderGraph
from torch_cluster.knn import knn_graph


####################
# model architecture 
####################

#########
# encoder
#########

# inputs are X_current and X_prev for last 5 time steps
# update edge matrix based on how close nodes are to each other
# E_v: MLP that converts each node in X_curr to latent vector size 128
# E_e: MLP that converts each node in E_curr to latent vector size 128 

class NodeEncoder(nn.Module):
    """
    Node encoder with:
    - 2 MLP layers with ReLU activations
    - output layer with no activation
    - LayerNorm on every layer

    Used to encode the node attributes to a latent space.

    Attributes:
    -----------
    node_encoder_stack : nn.Sequential
        The node encoder model.

    Methods: 
    --------
    __init__(n_features, n_latent)
        initializes model
    forward(X)
        computes the forward pass for the model.

    """
    def __init__(self, n_features, n_latent):
        """
        Initializes the model.

        Parameters:
        -----------
        n_features : int
            the number of unique features in one line in the trajectory
        n_latent : int
            the size of the latent space to be used in the hidden layers of the model

        Returns:
        --------
        """
        super(NodeEncoder, self).__init__()

        # here we consider n_nodes as a minibatch dimension, i.e. each batch member is 1 x n_features
        self.node_encoder_stack = nn.Sequential(
            nn.Linear(n_features, n_latent),
            nn.ReLU(),
            nn.LayerNorm([n_latent]), 
            nn.Linear(n_latent, n_latent), 
            nn.ReLU(),
            nn.LayerNorm([n_latent]),
            nn.Linear(n_latent, n_latent), 
            nn.LayerNorm([n_latent])
        )

    def forward(self, X):
        """
        Computes the forward pass for the model.

        Parameters:
        -----------
        X : Tensor
            node attributes of shape [n_nodes, n_features]

        Returns:
        --------
        X_h : Tensor
            latent representation of node attributes of shape [n_nodes, n_latent]
        """
        # in the relative version of the GNN, we mask the position, orientation and velocity data
        X[:, 1:] = 0
        # print("X with entries masked by 0s = ", X)

        X_h = self.node_encoder_stack(X)
        return X_h

class EdgeEncoder(nn.Module):
    """
    Edge encoder for absolute variant of model implementation. In the relative variant, we build a similar model to the NodeEncoder.

    Attributes:
    -----------
    n_latent : int
        The size of the latent space.
    edge_encoder_stack : nn.Linear
        The edge encoder model.

    Methods:
    --------
    __init__(n_edges, n_latent)
        Initializes the model
    forward(edge_attr)
        Implements the forward pass of the model
    """
    def __init__(self, n_features, n_latent):
        """
        Initializes the model.

        Parameters:
        -----------
        n_edges : int
            The number edges for the input graph
        n_latent : int
            The size of the latent space

        Returns:
        --------

        """
        super(EdgeEncoder, self).__init__()

        self.n_latent = n_latent

        # here we consider n_edges as a minibatch dimension, i.e. each batch member is 1 x n_features-1
        # we subtract 1 from features because the edge_attr matrix does not include nucleotide information
        self.edge_encoder_stack = nn.Sequential(
            nn.Linear(n_features-1, n_latent),
            nn.ReLU(),
            nn.LayerNorm([n_latent]), 
            nn.Linear(n_latent, n_latent), 
            nn.ReLU(),
            nn.LayerNorm([n_latent]),
            nn.Linear(n_latent, n_latent), 
            nn.LayerNorm([n_latent])
        )

    def forward(self, edge_attr):
        """
        Performs the forward pass for the model.

        Parameters:
        -----------
        edge_attr : Tensor
            edge attributes in shape [n_edges, n_features_edges]

        Returns:
        --------
        edge_attr_h : Tensor
            latent representation of edges of size [n_edges, n_latent]
        """
        edge_attr_h = self.edge_encoder_stack(edge_attr)
        return edge_attr_h

class Encoder(nn.Module):
    """
    Combines the node and edge encoders into one object. 

    Attributes:
    -----------
    node_encoder : NodeEncoder
        an instance of the NodeEncoder class
    edge_encoder : EdgeEncoder
        an instance of the EdgeEncoder class
    
    Methods:
    --------
    __init__(n_edges, n_features, n_latent)
        Initializes the model.
    forward(X, edge_attr)
        Performs the forward step for the encoder model.
    """
    def __init__(self, n_features, n_latent):
        """
        Initializes the node and edge encoders.

        Parameters:
        -----------
        n_edges : int
            The number edges for the input graph
        n_features : int
            the number of unique features in one line in the trajectory
        n_latent : int
            The size of the latent space
        
        Returns:
        --------
        """
        super(Encoder, self).__init__()

        self.node_encoder = NodeEncoder(n_features, n_latent)
        self.edge_encoder = EdgeEncoder(n_features, n_latent)

    def forward(self, X, edge_attr):
        """
        Implements the forward step of the model.

        Parameters:
        -----------
        X : Tensor
            node attributes of shape [n_nodes, n_features]
        edge_attr : Tensor
            edge attributes in shape [n_edges, n_features_edges]
        
        Returns:
        --------
        X_h : Tensor
            latent representation of node attributes of shape [n_nodes, n_latent]
        edge_attr_h : Tensor
            latent representation of edges of size [n_edges, n_latent]
        """
        X_h = self.node_encoder(X)
        edge_attr_h = self.edge_encoder(edge_attr)

        return X_h, edge_attr_h

###########
# processor
###########

# inputs are the latent graphs from encoder
# a set of K graph network layers 
# unshared parameters
# update functions are MLPs
# skip connections between input and output layers
# try PyTorch Geometric MetaLayer
# ref: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.meta.MetaLayer

class NodeModel(nn.Module):
    """
    The NodeModel is an MLP that processes the latent representation of the node attributes based on the graph connectivity.

    Attributes:
    -----------
    node_mlp_1 : nn.Sequential
        first layer that takes in X, E concatenated
    node_mlp_2 : nn.Sequential
        second layer that processes X, E and u (global parameters) concatenated
    node_mlp_3 : nn.Sequential
        output layer 

    Methods:
    --------
    __init__(n_latent)
        Initializes the model
    forward(x_h, edge_index, edge_attr_h, u, batch)
        Conducts the forward pass of the model.
    """
    def __init__(self, n_latent):
        """
        Initializes the NodeModel class. 

        Parameters:
        -----------
        n_latent : int
            The size of the latent space

        Returns:
        --------
        """
        super(NodeModel, self).__init__()

        # first layer handles X and E concatenated
        n_latent_1 = 2*n_latent
        self.node_mlp_1 = nn.Sequential(
            nn.Linear(n_latent_1, n_latent),
            nn.ReLU(),
            nn.LayerNorm([n_latent])
        )

        # second layer handles X, E and u (global) concatenated
        n_latent_2 = n_latent*2 + 1
        self.node_mlp_2 = nn.Sequential(
            nn.Linear(n_latent_2, n_latent),
            nn.ReLU(),
            nn.LayerNorm([n_latent])
        )
        
        # third layer is just an output layer, no change in dimensions
        self.node_mlp_3 = nn.Sequential(
            nn.Linear(n_latent, n_latent),
            nn.LayerNorm([n_latent])
        )

    def forward(self, x_h, edge_index, edge_attr_h, u, batch):
        """
        Implements the forward pass for the node model.

        Parameters: 
        -----------
        x_h : Tensor
            Size [n_nodes, n_latent], contains latent representation of nodes.
        edge_index : Tensor
            Size [2, n_edges] with max entry n_nodes - 1, provides the edges in COO format (each tuple is i,j index of edge in adjacency matrix)
        edge_attr_h : Tensor
            latent representation of edges of size [n_edges, n_latent]
        u : Tensor
            Size [B, F_u] -- these are global parameters (we do not use currently)
        batch : Tensor
            [N] with max entry B - 1. -- this lists which graph the nodes belong to (if more than one graph is contained in a batch)

        Returns:
        --------
        out : Tensor
            node attribute matrix of shape [n_nodes, n_latent]
        """
        
        row, col = edge_index
        out = torch.cat([x_h[row], edge_attr_h], dim=1)
        out = self.node_mlp_1(out)
        out = scatter_mean(out, col, dim=0, dim_size=x_h.size(0))
        out = torch.cat([x_h, out, u[batch]], dim=1)
        out = self.node_mlp_2(out)
        out = self.node_mlp_3(out)

        # add skip connection
        out += x_h
        return out

class EdgeModel(nn.Module):
    """
    The EdgeModel is an MLP that processes the latent representation of the edge attributes based on the graph connectivity.

    Attributes:
    -----------
    edge_mlp_1 : nn.Sequential
        An MLP that takes in the source, destination nodes and edge attributes
    edge_mlp_2 : nn.Sequential
        An MLP that takes in the data from the first MLP
    edge_mlp_3 : nn.Sequential
        An MLP that takes in the data from the second MLP

    Methods:
    --------
    __init__(n_edges, n_latent)
        Initializes the EdgeModel
    forward(src, dest, edge_attr_h, u, batch)
        Performs the forward pass for the model
    """
    def __init__(self, n_edges, n_latent):
        """
        Initializes an instance of the EdgeModel.

        Parameters:
        -----------
        n_edges : int
            The number edges for the input graph
        n_latent : int
            The size of the latent space
    
        Returns:
        --------

        """
        super(EdgeModel, self).__init__()
        
        # first layer handles source, destination nodes and edge attributes concatenated
        n_latent_1 = 3*n_latent + 1
        self.edge_mlp_1 = nn.Sequential(
            nn.Linear(n_latent_1, n_latent),
            nn.ReLU(),
            nn.LayerNorm([n_latent])
        )
        # second layer has no change in dimensions, no additional information provided
        self.edge_mlp_2 = nn.Sequential(
            nn.Linear(n_latent, n_latent),
            nn.ReLU(),
            nn.LayerNorm([n_latent])
        )
        # third layer is just an output layer, no change in dimensions
        self.edge_mlp_3 = nn.Sequential(
            nn.Linear(n_latent, n_latent),
            nn.LayerNorm([n_latent])
        )

    def forward(self, src, dest, edge_attr_h, u, batch):
        """
        Implements the forward pass of the edge model. 

        Parameters: 
        -----------
        src, dest : Tensors
            Tensors of size [n_edges, n_features]
        edge_attr_h : Tensor
            Size [n_edges, n_latent] containing edge attributes encoded into the latent space
        u : Tensor
            Size [B, F_u], where B is the number of graphs. --> not currently used
        batch : Tensor
            Size [n_edges] with max entry B - 1. Indicates which graph the edges belong to - they're all in the same batch in this implementation.

        Returns: 
        --------
        out : Tensor
            edge attribute matrix of size [n_edges, n_features_edges]
        """
        out = torch.cat([src, dest, edge_attr_h, u[batch]], 1)
        out = self.edge_mlp_1(out)
        out = self.edge_mlp_2(out)
        out = self.edge_mlp_3(out)

        out += edge_attr_h        
        return out


class Processor(nn.Module):
    """
    M-layers of a graph network. Start with 2.

    Attributes:
    -----------
    gn1, gn2 : MetaLayers
        The graph network layers implemented using PyTorch Geometric MetaLayers
    batch : Tensor
        This tensor assigns all the nodes to the correct graph - all nodes are part of the same graph. 
    u : Tensor
        This contains any global parameters we want to include - currently kept to 1.

    Methods:
    --------
    __init__(n_nodes, n_edges, n_latent)
        Initializes the Processor.
    forward(X_h, edge_index, edge_attr_h)
        Implements the forward pass for the Processor model.
    """
    def __init__(self, n_nodes, n_edges, n_latent):
        """
        Initializes an instance of the Processor.

        Parameters:
        -----------
        n_nodes : int
            The number of nodes in the input graph
        n_edges : int
            The number edges for the input graph
        n_latent : int
            The size of the latent space
    
        Returns:
        --------
        """
        super(Processor, self).__init__()

        self.gn1 = MetaLayer(EdgeModel(n_edges, n_latent), NodeModel(n_latent), None)
        self.gn2 = MetaLayer(EdgeModel(n_edges, n_latent), NodeModel(n_latent), None)

        self.batch = torch.zeros((n_nodes), dtype=torch.long) # batch assigns all nodes to the same graph
        self.u = torch.ones((1, 1)) # global attributes

    def forward(self, X_h, edge_index, edge_attr_h):
        """
        Implements the forward step for the Processor.

        Parameters:
        -----------
        X_h : Tensor
            Size [n_nodes, n_latent], contains latent representation of nodes.
        edge_index : Tensor
            Size [2, n_edges] with max entry n_nodes - 1, provides the edges in COO format (each tuple is i,j index of edge in adjacency matrix)
        edge_attr_h : Tensor
            latent representation of edges of size [n_edges, n_latent]

        Returns:
        --------
        X_m : Tensor
            The processed latent representation of the node attribute matrix.
        edge_attr2 : Tensor
            The processed latent representation of the edge attributes
        u2 : Tensor
            The processed latent representation of the global properties used in the model.
        """
        x1, edge_attr1, u1 = self.gn1(X_h, edge_index, edge_attr_h, self.u, self.batch)
        X_m, edge_attr2, u2 = self.gn2(x1, edge_index, edge_attr1, u1, self.batch)

        return X_m, edge_attr2, u2

        # op = MetaLayer(None, NodeModel(n_latent), None)
        # x, edge_attr, u = op(output, edge_index, edge_attr=edge_attr, u=u, batch=batch)


#########
# decoder
######### 

# takes final latent graph from processor
# MLP that outputs Y_hat, the accelerations in translation and rotation
# do NOT use LayerNorm on output of decoder
class Decoder(nn.Module):
    """
    Decoder with:
    - 2 MLP layers with ReLU activations
    - output layer with no activation
    - LayerNorm on every layer EXCEPT output layer

    Attributes:
    -----------
    decoder_stack : nn.Sequential
        The Decoder model.
    
    Methods:
    --------
    __init__(n_latent, Y_features)
        Initializes an instance of the Decoder.
    forward(X)
        Performs the forward step of the Decoder's computation. 
    """
    def __init__(self, n_latent, Y_features):
        """
        Initializes an instance of the Decoder.

        Parameters:
        -----------
        n_latent : int
            The size of the latent space
        Y_features : int
            the number of unique features in the output, y, which is the acceleration information for each node
        
        Returns:
        --------
        """
        super(Decoder, self).__init__()

        # here we consider n_nodes as a minibatch dimension, i.e. each batch member is 1 x n_features
        self.decoder_stack = nn.Sequential(
            nn.Linear(n_latent, n_latent),
            nn.ReLU(),
            nn.LayerNorm([n_latent]), 
            nn.Linear(n_latent, n_latent), 
            nn.ReLU(),
            nn.LayerNorm([n_latent]),
            nn.Linear(n_latent, Y_features)
            )

    def forward(self, X):
        """
        Performs the forward pass for the Processor. 

        Parameters:
        -----------
        X : Tensor
            node attributes of shape [n_nodes, n_features]

        Returns:
        --------
        Y : Tensor
            accelerations in translation and rotation in matrix of shape [n_nodes, Y_features]
        """
        Y = self.decoder_stack(X)
        return Y

class GNN(nn.Module):
    """
    The full GNN model. 

    Attributes:
    -----------
    encoder_model : Encoder
        An instance of the Encoder class
    processor_model : Processor
        An instance of the Processor class
    decoder_model : Decoder
        An instance of the Decoder class

    Methods:
    --------
    __init__(n_nodes, n_edges, n_features, n_latent, Y_features)
        Initializes an instance of the GNN class.
    forward(X, edge_index, edge_attr, dt, N, show_plot)
        Performs the forward step for the computation of the GNN
    rollout(rollout_steps, rollout_traj_file, t, top_file, traj_file, dt, N)
        Generates a novel rollout trajectory used to qualitatively verify how the model performs. 
    """
    def __init__(self, n_nodes, n_edges, n_features, n_latent, Y_features, gnd_time_interval): 
        """
        Initializes an instance of the GNN class.

        Parameters:
        -----------
        n_nodes : int
            The number of nodes in the input graph
        n_edges : int
            The number edges for the input graph
        n_features : int
            the number of unique features in one line in the trajectory
        n_latent : int
            The size of the latent space
        Y_features : int
            the number of unique features in the output, y, which is the acceleration information for each node
        gnd_time_interval : int
            Time represented by one time step in the ground truth data
        
        Returns:
        --------
        """
        self.gnd_time_interval = gnd_time_interval

        super(GNN, self).__init__()
        self.encoder_model = Encoder(n_features, n_latent)
        self.processor_model = Processor(n_nodes, n_edges, n_latent)
        self.decoder_model = Decoder(n_latent, Y_features)

    def forward(self, X, edge_index, edge_attr, dt, N=100, show_plot=False): 
        """
        Perform one step of the forward computation for the GNN.

        Parameters:
        -----------
        X : Tensor
            node attributes of shape [n_nodes, n_features]
        edge_index : Tensor
            Size [2, n_edges] with max entry n_nodes - 1, provides the edges in COO format (each tuple is i,j index of edge in adjacency matrix)
        edge_attr : Tensor
            edge attributes in shape [n_edges, n_features_edges]
        dt : int
            size of time step (in simulation units)
        N : int
            number of nodes
        show_plot : Boolean
            Flag used to signal if plots of loss curves should be shown during training run

        Return:
        -------
        y_h : Tensor
            The prediction of the accelerations Y
        X_next : Tensor
            The predicted next node attribute matrix X
        """

        # --- encoder ---
        X_h, edge_attr_h = self.encoder_model(X, edge_attr)

        # --- processor --- 
        X_m, _, _ = self.processor_model(X_h, edge_index, edge_attr_h)

        # --- decoder ---
        y_h = self.decoder_model(X_m)

        # --- update function ---
        X_next = doUpdate(X, y_h, dt, self.gnd_time_interval)

        # --- loss function ---
        # the loss function needs to compare the predicted acceleration with the target acceleration for a randomly selected set of nucleotides
        # generate a list of N randomly selected indices of nucleotides
        # N = 100 for a starting point
        # must generate random integers within 0 and X.shape[0] (i.e. n_nodes)
        # rand_idx = torch.randint(low=0, high=X.shape[0], size=(N,))

        # use Y, the predicted accelerations for those nucleotides
        # preds = Y[rand_idx]
        
        # return rand_idx, preds, X_next
        return y_h, X_next

    def rollout(self, k, X_norm, mean, std, rollout_steps, rollout_traj_file, t, top_file, traj_file, dt, N, n_features):
        """
        This function computes the trajectory for a given structure (defined by traj_file, top_file) for some time steps rollout_steps. Does not compare to ground truth - used to evaluate model's prediction capabilities. Rollout is saved to file.

        Parameters:
        -----------
        k : int
            Number of neighbors to consider
        X_norm : Tensor
            Normalized X node attribute matrix.
        mean : Tensor
            Mean for each column in X
        std : Tensor
            Standard deviation for each column in X
        rollout_steps : int
            The number of time steps to generate a trajectory for.
        rollout_traj_file : str
            Filename used to save the rollout
        t : int
            Current time step
        top_file : str
            string containing full address of topology file (.top)      
        traj_file : str
            full address of trajectory file (.dat)
        dt : int
            size of time step (in simulation units)
        N : int
            number of nodes

        Returns: 
        --------
        """
       
        with torch.no_grad():
            _, E_backbone = makeGraphfromTraj(top_file, traj_file, N)

            # get normalized X from dataset class, reverse normalization
            X_unnorm = reverseNormalizeX(X_norm, mean, std)

            # save unnormalized X to file 
            with open(rollout_traj_file, "w") as f:
                f.write("t = {0}\n".format(t))
                f.write("b = 10 10 10\n")
                f.write("E = 0 0 0\n")

                X_np = X_unnorm.numpy()
                for i in range(X_np.shape[0]):
                    my_str = ""
                    for j in range(1, X_np.shape[1]):
                        my_str += str(X_np[i,j])
                        my_str += " "
                    my_str += "\n"
                    f.write(my_str)

            for z in tqdm(range(rollout_steps)):

                # build up the adjacency matrix, E, for this time step
                # compute E_neighbors by providing X[:,1:3] to knn_graph and asking for k nearest neighbors
                output = knn_graph(X_norm[:,1:4], k, flow="target_to_source")
                row = output[0,:]
                col = output[1,:]
                data = torch.ones_like(row)
                coo = coo_matrix((data, (row, col)), shape=(X_norm.shape[0], X_norm.shape[0]))
                E_knn = torch.from_numpy(coo.todense())
                
                # combines the backbone edges and knn edges
                E = E_knn + E_backbone

                # convert the output to a coo-matrix
                edges_coo = coo_matrix(E)
                edge_attr = np.array([edges_coo.data], dtype=np.int_)
                edge_index = np.array([[edges_coo.row], [edges_coo.col]], dtype=np.int_)
                edge_index = np.reshape(edge_index, (edge_index.shape[0], edge_index.shape[2]))

                # define edge attributes using edges_coo
                # so every row contains the relative position, orientation and velocity data between node i and j

                # initialize an empty data matrix of size [n_edges, n_features-1]
                n_edges = edge_index.shape[1]
                edge_attr = torch.zeros(n_edges, n_features-1)

                # iterate through every edge
                for i in range(n_edges):

                    # get the i-th and j-th nodes' indices
                    idx_i = edges_coo.row[i]
                    idx_j = edges_coo.col[i]

                    # get the i-th and j-th nodes' X data
                    node_i = X_norm[idx_i, 1:]
                    node_j = X_norm[idx_j, 1:]

                    # compute the difference between them
                    delta_X = node_i - node_j

                    # add that difference vector to the data matrix in the corresponding row
                    edge_attr[i] = delta_X

                # convert to torch tensors
                edge_index = torch.from_numpy(edge_index)

                # # convert to torch tensors
                # edge_index = torch.from_numpy(edge_index)
                # edge_attr = torch.from_numpy(edge_attr.T)

                t += dt
                _, X_next = self(X_norm, edge_index, edge_attr, dt, N=N) 

                X_next_unnorm = reverseNormalizeX(X_next, mean, std)

                # save the unnormalized X_next to file
                with open(rollout_traj_file, "a") as f:
                    f.write("t = {0}\n".format(t))
                    f.write("b = 84.160285949707 84.160285949707 84.160285949707\n")
                    f.write("E = 0 0 0\n")

                    X_next_np = X_next_unnorm.numpy()
                    for i in range(X_next_np.shape[0]):
                        my_str = ""
                        for j in range(1, X_next_np.shape[1]):
                            my_str += str(X_next_np[i,j])
                            my_str += " "
                        my_str += "\n"
                        f.write(my_str)

                X_norm = X_next

