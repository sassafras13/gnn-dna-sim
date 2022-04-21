# Objective: given input data, build node vector and edge matrix

import matplotlib.pyplot as plt
import numpy as np
from model import GNN
import torch
from torch import nn
from torch_geometric.nn import MetaLayer
from torch_scatter import scatter_mean
from tqdm import tqdm
from utils import makeGraphfromTraj, plotGraph, doUpdate, getGroundTruthY


def main():
    # --- define important parameters ---
    n_features = 16 
    n_latent = 128
    Y_features = 6
    dt = 100 # time steps from sim_out step
    t = 100 # track the current time step
    tf = 50000 # final time step
    # n_timesteps = int(tf / dt)
    n_timesteps = 5
    epochs = 1 
    lr = 1e-4
    show_plot = False
    loss_list = np.zeros((n_timesteps,))

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
    # traj_file = "/home/emma/Documents/Classes/10-707/final-project/wireframe-dataset/2D/hexagon/relax_out/trajectory_relax.dat"

    # --- build the initial graph ---
    X, E = makeGraphfromTraj(top_file, traj_file)
    print("X shape, ", X.shape)
    print("E shape, ", E.shape)

    # --- plot the graph ---
    if show_plot == True:
        plotGraph(X,E)

    # --- model ---
    model = GNN(n_features, n_latent, Y_features)

    # --- loss function ---
    loss_fn = nn.MSELoss()

    # --- optimizer ---
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # --- training loop ---
    for i in range(epochs): 

        for j in tqdm(range(n_timesteps)):

            rand_idx, preds, X_next = model(X, E, dt, N=100) # TODO: Modify to take in X and E, not the input files

            target = getGroundTruthY(traj_file, t, dt, X_next, rand_idx)

            loss = loss_fn(preds, target)
            loss_list[j] = loss.item()

            # print("loss", loss)

            # --- backpropagation ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # --- update the graph for the next time step
            X = X_next 
            X = X.detach_() # removes the tensor from the computational graph - it is now a leaf
            # TODO: What to do about updating E? 

            # update the time 
            t += dt

    plt.plot(list(range(n_timesteps)), loss_list)
    plt.xlabel("Time step")
    plt.ylabel("Loss")
    plt.title("Loss curve for 1 epoch")
    plt.show()

    # --- rollout a trajectory ---
    rollout_steps = 5
    rollout_traj_file = "/home/emma/Documents/Classes/10-707/final-project/wireframe-dataset/2D/hexagon/rollout.dat"
    t0 = 100

    model.rollout(rollout_steps, rollout_traj_file, t0, top_file, traj_file, dt)

if __name__ == "__main__":
    main()
