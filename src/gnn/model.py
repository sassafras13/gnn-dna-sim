import numpy as np
from scipy.sparse import coo_matrix
import torch
from torch import nn
from torch_geometric.nn import MetaLayer
from torch_scatter import scatter_mean

# select device to use
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))

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
        output = self.node_encoder_stack(X)
        return output


# class EdgeEncoder():
#     """
#     Edge encoder with:
#     - 2 MLP layers with ReLU activations
#     - output layer with no activation
#     - LayerNorm on every layer

#     Input:
#     E : edge attributes of shape [n_nodes, n_nodes]

#     Output:
#     E_h : latent representation of node attributes of shape [n_nodes, n_latent]
#     """
#     def __init__(self, n_features, n_latent):
#         super(EdgeEncoder, self).__init__()

#         # here we consider n_nodes as a minibatch dimension, i.e. each batch member is 1 x n_features
#         self.edge_encoder_stack = nn.Sequential(
#             nn.Linear(n_features, n_latent),
#             nn.ReLU(),
#             nn.LayerNorm([n_latent]), 
#             nn.Linear(n_latent, n_latent), 
#             nn.ReLU(),
#             nn.LayerNorm([n_latent]),
#             nn.Linear(n_latent, n_latent), 
#             nn.LayerNorm([n_latent])
#         )

#     def forward(self, X):
#         output = self.edge_encoder_stack(X)
#         return output

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
        print("size of edge_attr", edge_attr.shape)
        x1, edge_attr1, u1 = self.gn1(output, edge_index, edge_attr, u, batch)
        x2, edge_attr2, u2 = self.gn2(x1, edge_index, edge_attr1, u1, batch)

        return x2, edge_attr2, u2

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
    X_h : latent representation of node attributes of shape [n_nodes, n_latent]
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
        output = self.decoder_stack(X)
        return output

# def main():

# if __name__ == "__main__":
#     main()