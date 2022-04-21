import numpy as np
from scipy.sparse import coo_matrix
import torch
from torch import nn
from torch_geometric.nn import MetaLayer
from torch_scatter import scatter_mean
from tqdm import tqdm
from utils import makeGraphfromTraj, plotGraph, doUpdate

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

###########
# processor
###########

# inputs are the latent graphs from encoder
# a set of K graph network layers 
# shared or unshared parameters
# update functions are MLPs
# skip connections between input and output layers
# try PyTorch Geometric MetaLayer
# ref: https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.meta.MetaLayer

class NodeModel(nn.Module):
    def __init__(self, n_latent):
        super(NodeModel, self).__init__()
        n_latent_1 = n_latent + 1
        self.node_mlp_1 = nn.Sequential(
            nn.Linear(n_latent_1, n_latent),
            nn.ReLU(),
            nn.LayerNorm([n_latent])
        )
        n_latent_2 = n_latent*2 + 1
        self.node_mlp_2 = nn.Sequential(
            nn.Linear(n_latent_2, n_latent),
            nn.ReLU(),
            nn.LayerNorm([n_latent])
        )

    def forward(self, x, edge_index, edge_attr, u, batch):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1, and E is the number of edges
        # edge_attr: [E, F_e]
        # u: [B, F_u] -- these are global parameters
        # batch: [N] with max entry B - 1. -- this lists which graph the nodes belong to (if more than one graph is contained in a batch)
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out, u[batch]], dim=1)
        return self.node_mlp_2(out)

class Processor(nn.Module):
    """
    M-layers of a graph network. Need M = 10 ideally, but maybe start with 2 and play with this later.

    """
    def __init__(self, n_latent):
        super(Processor, self).__init__()

        self.gn1 = MetaLayer(None, NodeModel(n_latent), None)
        self.gn2 = MetaLayer(None, NodeModel(n_latent), None)

    def prepareGraphForProcessor(self, E, output):
        """
        Generates required variables that describe the graph G = (X, E) for use with processor.

        Inputs: 
        X : node attributes of shape [n_nodes, n_features]
        E : edge attributes/adjacency matrix of shape [n_nodes, n_nodes]

        Outputs: 
        edge_attr : edge attributes in shape [n_edges, n_features_edges]
        edge_index : row and column indices for edges in COO format in matrix of shape [2, n_edges]
        batch : list of indices indicating which graph each node belongs to, given in matrix shape []
        u : global attributes (currently set to 1)
        """
        # need to convert adjacency matrix E to edge_index in COO format
        edge_index_coo = coo_matrix(E)
        edge_attr = np.array([edge_index_coo.data], dtype=np.int_)
        edge_index = np.array([[edge_index_coo.row], [edge_index_coo.col]], dtype=np.int_)
        edge_index = np.reshape(edge_index, (edge_index.shape[0], edge_index.shape[2]))

        # convert to torch tensors
        edge_index = torch.from_numpy(edge_index)
        edge_attr = torch.from_numpy(edge_attr.T)
        # print("edge index size", edge_index.shape) # should be [2, E] 
        # print("edge attr size", edge_attr.shape) # should be [E, F_e] 
        
        batch = torch.zeros((output.shape[0]), dtype=torch.long) # batch assigns all nodes to the same graph
        u = torch.ones((1, 1)) # global attributes

        return edge_attr, edge_index, batch, u

    def forward(self, E, output):
        edge_attr, edge_index, batch, u = self.prepareGraphForProcessor(E, output)
        x1, edge_attr1, u1 = self.gn1(output, edge_index, edge_attr, u, batch)
        X_m, edge_attr2, u2 = self.gn2(x1, edge_index, edge_attr1, u1, batch)

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
    Y : accelerations in translation and rotation in matrix of shape [n_, Y_features]
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
        print("size of Y", Y.shape)
        return Y

class GNN(nn.Module):
    """
    The full model. 
    """
    def __init__(self, n_features, n_latent, Y_features): 
        super(GNN, self).__init__()
        self.encoder_model = NodeEncoder(n_features, n_latent)
        self.processor_model = Processor(n_latent)
        self.decoder_model = Decoder(n_latent, Y_features)

    def forward(self, X, E, dt, N=100, show_plot=False): 

        # --- encoder ---
        enc_output = self.encoder_model(X)

        # --- processor --- 
        proc_output, _, _ = self.processor_model(E, enc_output)

        # --- decoder ---
        Y = self.decoder_model(proc_output)

        # --- update function ---
        X_next = doUpdate(X, Y, dt)

        # --- loss function ---
        # the loss function needs to compare the predicted acceleration with the target acceleration for a randomly selected set of nucleotides
        # generate a list of N randomly selected indices of nucleotides
        # N = 100 for a starting point
        # must generate random integers within 0 and X.shape[0] (i.e. n_nodes)
        rand_idx = torch.randint(low=0, high=X.shape[0], size=(N,))

        # use Y, the predicted accelerations for those nucleotides
        preds = Y[rand_idx]
        
        return rand_idx, preds, X_next

    def rollout(self, rollout_steps, rollout_traj_file, t, top_file, traj_file, dt):
       
        with torch.no_grad():
            X, E = makeGraphfromTraj(top_file, traj_file)

            # save X to file 
            with open(rollout_traj_file, "w") as f:
                f.write("t = {0}\n".format(t))
                f.write("b = 84.160285949707 84.160285949707 84.160285949707\n")
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
                _, _, X_next = self(X, E, dt, N=100) 

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

