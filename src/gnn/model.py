import numpy as np
from scipy.sparse import coo_matrix
import torch
from torch import nn
from torch_geometric.nn import MetaLayer
from torch_scatter import scatter_mean
from tqdm import tqdm
from utils import makeGraphfromTraj, plotGraph, doUpdate, prepareEForModel
from data import DatasetGraph, DataloaderGraph

####################
# MLP baseline model
####################
class MlpModel(nn.Module):

    def __init__(self, n_features, n_latent, Y_features):
        super(MlpModel, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(n_features, n_latent),
            nn.ReLU(),
            nn.Linear(n_latent, n_latent),
            nn.ReLU(),
            nn.Linear(n_latent, Y_features)
        )

    def forward(self, X):
        y_h = self.mlp(X)
        return y_h
    
    def rollout(self, rollout_steps, rollout_traj_file, t, top_file, traj_file, dt, N):
       
        with torch.no_grad():

            X, _ = makeGraphfromTraj(top_file, traj_file, N)

            # save X to file 
            with open(rollout_traj_file, "w") as f:
                f.write("t = {0}\n".format(t))
                f.write("b = 10 10 10\n")
                f.write("E = 0 0 0\n")

                X_np = X.numpy()
                for i in range(X_np.shape[0]):
                    my_str = ""
                    for j in range(1, X_np.shape[1]):
                        my_str += str(X_np[i,j])
                        my_str += " "
                    my_str += "\n"
                    f.write(my_str)

            # generate the rollout
            for k in tqdm(range(rollout_steps)):
                t += dt
                y_h = self(X) 
                X_next = doUpdate(X, y_h, dt)

                # save the X_next to file
                with open(rollout_traj_file, "a") as f:
                    f.write("t = {0}\n".format(t))
                    f.write("b = 84.160285949707 84.160285949707 84.160285949707\n")
                    f.write("E = 0 0 0\n")

                    X_next_np = X_next.numpy()
                    for i in range(X_next_np.shape[0]):
                        my_str = ""
                        for j in range(1, X_next_np.shape[1]):
                            my_str += str(X_next_np[i,j])
                            my_str += " "
                        my_str += "\n"
                        f.write(my_str)

                X = X_next



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

    Input:
    X : node attributes of shape [n_nodes, n_features]

    Output:
    X_h : latent representation of node attributes of shape [n_nodes, n_latent]
    """
    def __init__(self, n_features, n_latent):
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
        X_h = self.node_encoder_stack(X)
        return X_h

class EdgeEncoder(nn.Module):
    """
    Edge encoder for absolute variant of model implementation. In the absolute variant, we simply return a bias vector of size n_latent.

    Input:
    edge_attr : edge attributes in shape [n_edges, n_features_edges]
    edge_index : row and column indices for edges in COO format in matrix of shape [2, n_edges]

    Output:
    edge_attr_h : latent representation of edges of size [n_edges, n_latent]
    """
    def __init__(self, n_edges, n_latent):
        super(EdgeEncoder, self).__init__()

        self.n_latent = n_latent

        # here we just use a linear layer of size [n_edges, n_latent] to get the bias vector
        self.edge_encoder_stack = nn.Linear(1, n_latent)

    def forward(self, edge_attr):
        # need to convert edge_attr to type float
        edge_attr = edge_attr.type(torch.float)

        # we are not actually going to use edge_attr directly in this implementation
        # instead we generate a set of zeros of the same size
        E_zero = torch.zeros_like(edge_attr, dtype=torch.float)
        # E_zero = torch.flatten(E_zero) # convert to size [n_edges * n_features_edges] = [n_edges, ]
        edge_attr_h = self.edge_encoder_stack(E_zero) # this should return a bias vector of size [n_edges, n_latent]

        # E_h = torch.reshape(E_h, (self.n_nodes * self.n_nodes, self.n_latent)) # convert to size [n_nodes*n_nodes, n_latent]

        # # we use E to mask entries in E_h where edges do not exist
        # E = torch.flatten(E) # convert to size [n_nodes*n_nodes]
        # E_mat = E.repeat(1,self.n_latent).reshape((self.n_latent, self.n_nodes*self.n_nodes)).T

        # # dot product to mask entries
        # E_h = E_h * E_mat

        # # reshape to size of original tensor, with expanded latent dimension
        # E_h = torch.reshape(E_h, (self.n_nodes, self.n_nodes, self.n_latent)) # convert to shape [n_nodes, n_nodes, n_latent]

        return edge_attr_h

class Encoder(nn.Module):
    def __init__(self, n_edges, n_features, n_latent):
        super(Encoder, self).__init__()

        self.node_encoder = NodeEncoder(n_features, n_latent)
        self.edge_encoder = EdgeEncoder(n_edges, n_latent)

    def forward(self, X, edge_attr):
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
    def __init__(self, n_latent):
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

        Inputs: 
        x_h : [n_nodes, n_latent], where N is the number of nodes.
        edge_index : [2, n_edges] with max entry N - 1, and E is the number of edges
        edge_attr_h : [n_edges, n_latent]
        u : [B, F_u] -- these are global parameters (we do not use currently)
        batch : [N] with max entry B - 1. -- this lists which graph the nodes belong to (if more than one graph is contained in a batch)

        Outputs:
        out : node attribute matrix of shape [n_nodes, n_latent]
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
    def __init__(self, n_edges, n_latent):
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

        Inputs: 
        src, dest : [n_edges, n_features], where E is the number of edges.
        edge_attr_h : [n_edges, n_latent]
        u : [B, F_u], where B is the number of graphs. --> not currently used
        batch : [n_edges] with max entry B - 1.

        Outputs: 
        out : edge attribute matrix of size [n_edges, n_features_edges]
        """
        out = torch.cat([src, dest, edge_attr_h, u[batch]], 1)
        out = self.edge_mlp_1(out)
        out = self.edge_mlp_2(out)
        out = self.edge_mlp_3(out)

        out += edge_attr_h        
        return out


class Processor(nn.Module):
    """
    M-layers of a graph network. Need M = 10 ideally, but maybe start with 2 and play with this later.

    batch : list of indices indicating which graph each node belongs to, given in matrix shape []
    u : global attributes (currently set to 1)
    """
    def __init__(self, n_nodes, n_edges, n_latent):
        super(Processor, self).__init__()

        self.gn1 = MetaLayer(EdgeModel(n_edges, n_latent), NodeModel(n_latent), None)
        self.gn2 = MetaLayer(EdgeModel(n_edges, n_latent), NodeModel(n_latent), None)

        self.batch = torch.zeros((n_nodes), dtype=torch.long) # batch assigns all nodes to the same graph
        self.u = torch.ones((1, 1)) # global attributes

    def forward(self, X_h, edge_index, edge_attr_h):
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

    Input:
    X : node attributes of shape [n_nodes, n_features]

    Output:
    Y : accelerations in translation and rotation in matrix of shape [n_nodes, Y_features]
    """
    def __init__(self, n_latent, Y_features):
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
        Y = self.decoder_stack(X)
        return Y

class GNN(nn.Module):
    """
    The full model. 
    """
    def __init__(self, n_nodes, n_edges, n_features, n_latent, Y_features): 
        super(GNN, self).__init__()
        self.encoder_model = Encoder(n_edges, n_features, n_latent)
        self.processor_model = Processor(n_nodes, n_edges, n_latent)
        self.decoder_model = Decoder(n_latent, Y_features)

    def forward(self, X, edge_index, edge_attr, dt, N=100, show_plot=False): 

        # --- encoder ---
        X_h, edge_attr_h = self.encoder_model(X, edge_attr)

        # --- processor --- 
        X_m, _, _ = self.processor_model(X_h, edge_index, edge_attr_h)

        # --- decoder ---
        y_h = self.decoder_model(X_m)

        # --- update function ---
        X_next = doUpdate(X, y_h, dt)

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

    def rollout(self, rollout_steps, rollout_traj_file, t, top_file, traj_file, dt, N):
       
        with torch.no_grad():
            X, E = makeGraphfromTraj(top_file, traj_file, N)
            edge_attr, edge_index = prepareEForModel(E)

            # save X to file 
            with open(rollout_traj_file, "w") as f:
                f.write("t = {0}\n".format(t))
                f.write("b = 10 10 10\n")
                f.write("E = 0 0 0\n")

                X_np = X.numpy()
                for i in range(X_np.shape[0]):
                    my_str = ""
                    for j in range(1, X_np.shape[1]):
                        my_str += str(X_np[i,j])
                        my_str += " "
                    my_str += "\n"
                    f.write(my_str)

            for k in tqdm(range(rollout_steps)):
                t += dt
                _, X_next = self(X, edge_index, edge_attr, dt, N=N) 

                # save the X_next to file
                with open(rollout_traj_file, "a") as f:
                    f.write("t = {0}\n".format(t))
                    f.write("b = 84.160285949707 84.160285949707 84.160285949707\n")
                    f.write("E = 0 0 0\n")

                    X_next_np = X_next.numpy()
                    for i in range(X_next_np.shape[0]):
                        my_str = ""
                        for j in range(1, X_next_np.shape[1]):
                            my_str += str(X_next_np[i,j])
                            my_str += " "
                        my_str += "\n"
                        f.write(my_str)

                X = X_next

