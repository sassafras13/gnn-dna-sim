# Objective: given input data, build node vector and edge matrix

import matplotlib.pyplot as plt
import numpy as np
from model import NodeEncoder, Processor, Decoder
import torch
from torch import nn
from torch_geometric.nn import MetaLayer
from torch_scatter import scatter_mean
from utils import makeGraph, plotGraph, doUpdate, getGroundTruthY


def main():
    # --- define important parameters ---
    n_features = 16 
    n_latent = 128
    Y_features = 6
    dt = 100 # time steps from sim_out step
    t = 100 # track the current time step

    # --- select device to use ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    # --- input data ---
    # topology file
    top_file = "/home/emma/Documents/Classes/10-707/final-project/wireframe-dataset/2D/hexagon/08_hexagon_DX_42bp-segid.pdb.top"

    # initial configuration
    config_file = "/home/emma/Documents/Classes/10-707/final-project/wireframe-dataset/2D/hexagon/08_hexagon_DX_42bp-segid.pdb.oxdna"

    # trajectory file
    traj_file = "/home/emma/Documents/Classes/10-707/final-project/wireframe-dataset/2D/hexagon/sim_out/trajectory_sim.dat"

    # --- build the graph ---
    X, E = makeGraph(top_file, config_file)
    print("X shape, ", X.shape)
    print("E shape, ", E.shape)
    print("X size, ", X.nbytes)
    print("E size, ", E.nbytes)

    # --- plot the graph ---
    # plotGraph(X,E)

    # --- encoder ---
    encoder_model = NodeEncoder(n_features, n_latent).to(device)
    # print(encoder_model)

    output = encoder_model.forward(torch.from_numpy(X).float())
    print("output size before processor", output.shape)

    # --- processor --- 
    processor_model = Processor(n_latent).to(device)
    # print(processor_model)

    x, edge_attr, u = processor_model.forward(E, output)
    # edge_attr, edge_index, batch, u = prepareGraphForProcessor(E, output)
    # op = MetaLayer(None, NodeModel(n_latent), None)
    # x, edge_attr, u = op(output, edge_index, edge_attr=edge_attr, u=u, batch=batch)

    print("x from metalayer shape", x.shape)

    # --- decoder ---
    decoder_model = Decoder(n_latent, Y_features).to(device)
    print(decoder_model)
    Y = decoder_model.forward(x)
    print("output size after decoder", Y.shape)

    # --- update function ---
    X_next = doUpdate(torch.from_numpy(X).float(), Y, dt)

    # --- loss function ---
    # the loss function needs to compare the predicted acceleration with the target acceleration for a randomly selected set of nucleotides
    # generate a list of N randomly selected indices of nucleotides
    # N = 100 for a starting point
    # must generate random integers within 0 and X.shape[0] (i.e. n_nodes)
    N = 100
    rand_idx = torch.randint(low=0, high=X.shape[0], size=(N,))

    # use Y, the predicted accelerations for those nucleotides
    preds = Y[rand_idx]
    print("size of preds", preds.shape)

    # extract X_t and X_t+1 from the training data for those nucleotides and use them to compute the accelerations
    target = getGroundTruthY(traj_file, t, dt, torch.from_numpy(X).float(), rand_idx)

    loss = nn.MSELoss()
    output = loss(preds, target)
    print("loss", output)


if __name__ == "__main__":
    main()
