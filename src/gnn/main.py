import matplotlib.pyplot as plt
import numpy as np
from model import GNN, EdgeEncoder
import torch
from torch import nn
from torch_geometric.nn import MetaLayer
from torch_scatter import scatter_mean
from tqdm import tqdm
from utils import makeGraphfromTraj, plotGraph, doUpdate, getGroundTruthY, prepareEForModel


def main():
    # --- define important parameters ---
    n_features = 16 
    n_latent = 128
    Y_features = 6
    dt = 100 # time steps from sim_out step
    t = 100 # track the current time step
    tf = 50000 # final time step
    # n_timesteps = int(tf / dt)
    n_timesteps = 10
    epochs = 1 
    lr = 1e-4
    show_plot = False
    show_rollout = False

    # --- variables for storing loss ---
    train_loss_list = np.zeros((epochs,n_timesteps))
    val_loss_list = np.zeros((epochs,n_timesteps))

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
    n_nodes = X.shape[0]

    edge_attr, edge_index = prepareEForModel(E)
    n_edges = edge_attr.shape[0]

    # --- plot the graph ---
    if show_plot == True:
        plotGraph(X,E)

    # --- model ---
    model = GNN(n_nodes, n_edges, n_features, n_latent, Y_features)

    # --- loss function ---
    loss_fn = nn.MSELoss()

    # --- optimizer ---
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # --- scheduler ---
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    # --- training loop ---
    for i in range(epochs): 

        # train
        print("--- Epoch {0} ---".format(i+1))
        model.train(True)
        train_t = t
        print("---- Train ----")
        for j in tqdm(range(n_timesteps)):
            print("t in train loop", train_t)

            rand_idx, preds, X_next = model(X, edge_index, edge_attr, dt, N=100) 

            target = getGroundTruthY(traj_file, t, dt, X_next, rand_idx)

            loss = loss_fn(preds, target)
            train_loss_list[i,j] = loss.item()

            # print("train loss", loss.item())

            # --- backpropagation ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # --- update the graph for the next time step
            X = X_next 
            X = X.detach_() # removes the tensor from the computational graph - it is now a leaf
            # TODO: What to do about updating E? 

            # update the time 
            train_t += dt
        
        scheduler.step() # reduce the learning rate after every epoch

        # validate
        model.train(False)
        valid_t = t
        print("---- Validate ----")
        for k in tqdm(range(n_timesteps)):
            print("t in valid loop", valid_t)
            rand_idx, preds, X_next = model(X, edge_index, edge_attr, dt, N=100) 

            target = getGroundTruthY(traj_file, t, dt, X_next, rand_idx)

            loss = loss_fn(preds, target)
            val_loss_list[i,k] = loss.item()

            # print("validation loss", loss.item())

            # --- update the graph for the next time step
            X = X_next 
            X = X.detach_() # removes the tensor from the computational graph - it is now a leaf

            # update the time 
            valid_t += dt


        # plot the loss curves for that epoch
        plt.plot(list(range(n_timesteps)), train_loss_list[i,:], "-k", label="Train")
        plt.plot(list(range(n_timesteps)), val_loss_list[i,:], "-r", label="Validate")
        plt.xlabel("Time step")
        plt.ylabel("Loss")
        plt.title("Loss curve for {0}th epoch".format(i))
        plt.legend()
        plt.savefig("/home/emma/Documents/Classes/10-707/final-project/wireframe-dataset/2D/hexagon/loss_curves_epoch_{0}.png".format(i))
        plt.clf()

    # plot the loss curves for all epochs
    plt.plot(list(range(epochs)), train_loss_list[:,-1], "-k", label="Train")
    plt.plot(list(range(epochs)), val_loss_list[:,-1], "-r", label="Validate")
    plt.xlabel("Time step")
    plt.ylabel("Loss")
    plt.title("Loss curve for all epochs")
    plt.legend()
    plt.savefig("/home/emma/Documents/Classes/10-707/final-project/wireframe-dataset/2D/hexagon/loss_curves_all_epochs.png")

    # --- rollout a trajectory ---
    if show_rollout == True:
        rollout_steps = 5
        rollout_traj_file = "/home/emma/Documents/Classes/10-707/final-project/wireframe-dataset/2D/hexagon/rollout.dat"
        t0 = 100

        model.rollout(rollout_steps, rollout_traj_file, t0, top_file, traj_file, dt)

if __name__ == "__main__":
    main()
