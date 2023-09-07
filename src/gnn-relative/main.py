import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
from model import GNN
from data import DatasetGraph, DataloaderGraph
import torch
from torch import nn
from torch_geometric.nn import MetaLayer
from torch_scatter import scatter_mean
from tqdm import tqdm
import wandb
import random

from utils import makeGraphfromTraj, plotGraph, getGroundTruthY, prepareEForModel, getForcesandTorques, sim2RealUnits, buildX

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Training GNN for DNA"
    )
    parser.add_argument("--n_nodes", type=int, default=40, help="Number of nodes in graph")
    parser.add_argument("--n_edges", type=int, default=20, help="Number of edges in graph")
    parser.add_argument("--n_features", type=int, default=16, help="Number of node features in input data")
    parser.add_argument("--n_latent", type=int, default=128, help="Number of latent features to encode node and edge attributes")
    parser.add_argument("--Y_features", type=int, default=6, help="Number of state variables output by model")
    parser.add_argument("--dt", type=int, default=100, help="Time interval in input data")
    parser.add_argument("--tf", type=int, default=99900, help="Final time step of input data")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--show_plot", type=bool, default=True, help="Show plot of initial structure")
    parser.add_argument("--show_rollout", type=bool, default=True, help="Generate a rollout trajectory")
    parser.add_argument("--rollout_steps", type=int, default=10, help="Number of steps in rollout trajectory")
    parser.add_argument("--top_file", type=str, default="/home/emma/repos/gnn-dna-sim/src/dataset-generation/dsDNA/top.top", help="Address of topology file")
    parser.add_argument("--traj_file", type=str, default="/home/emma/Documents/research/gnn-dna/dsdna-dataset/training/trajectory_sim_traj1.dat", help="Address of an example trajectory file to help generate rollout")
    parser.add_argument("--train_dir", type=str, default="/home/emma/Documents/research/gnn-dna/dsdna-dataset/training/", help="Location of training data")
    parser.add_argument("--val_dir", type=str, default="/home/emma/Documents/research/gnn-dna/dsdna-dataset/validation/", help="Directory containing validation dataset")
    parser.add_argument("--checkpoint_period", type=int, default=1, help="Interval between saving checkpoints during training")
    parser.add_argument("--n_train", type=int, default=8, help="Number of training trajectories")
    parser.add_argument("--n_val", type=int, default=2, help="Number of validation trajectories")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--k", type=int, default=3, help="Number of neighbors to include in adjacency matrix.")
    parser.add_argument("--noise_std", type=float, default=0.003, help="Std deviation of noise to add to training data.")
    parser.add_argument("--gnd_time_interval", type=float, default=0.005, help="Time interval for one time step in original simulation tool.")
    args, unknown = parser.parse_known_args()
    return args

def main(args):

    # --- define important parameters ---
    n_nodes = args.n_nodes
    n_edges = args.n_edges
    n_features = args.n_features
    n_latent = args.n_latent
    Y_features = args.Y_features
    dt = args.dt # time steps from sim_out step
    t = 100 # track the current time step
    tf = args.tf # final time step
    n_timesteps = int(tf / dt)
    epochs = args.epochs
    lr = args.lr
    show_plot = args.show_plot
    show_rollout = args.show_rollout
    rollout_steps = args.rollout_steps
    top_file = args.top_file
    traj_file = args.traj_file
    train_dir = args.train_dir
    val_dir = args.val_dir
    checkpoint_period = args.checkpoint_period
    n_train = args.n_train
    n_val = args.n_val
    seed = args.seed
    k = args.k
    noise_std = args.noise_std
    gnd_time_interval = args.gnd_time_interval

    # start a new wandb run to track this script
    run = wandb.init(
            # set the wandb project where this run will be logged
            project="GNN+DNA",
            
            # track hyperparameters and run metadata
            config={
            "n_nodes" : n_nodes,
            "n_edges" : n_edges,
            "n_features" : n_features,
            "n_latent" : n_latent,
            "Y_features" : Y_features,
            "dt" : dt,
            "tf" : tf,  
            "n_timesteps" : n_timesteps,
            "epochs" : epochs,
            "lr" : lr,
            "rollout_steps" : rollout_steps,
            "top_file" : top_file,
            "traj_file" : traj_file,
            "n_train" : n_train,
            "n_val" : n_val,
            "k" : k,
            "noise_std" : noise_std,
            "gnd_time_interval" : gnd_time_interval
            }
    )

    # --- variables for storing loss ---
    train_loss_list = np.zeros((epochs, n_train, n_timesteps)) # epochs, n_train files, n_timesteps
    val_loss_list = np.zeros((epochs, n_val, n_timesteps)) # epochs, n_validation files, n_timesteps

    # --- select device to use ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    # --- init dataset and dataloader classes ---
    train_dataset = DatasetGraph(train_dir, n_nodes, n_features, dt, n_timesteps, k, gnd_time_interval, noise_std)
    train_dataloader = DataloaderGraph(train_dataset, n_timesteps, shuffle=True)
    val_dataset = DatasetGraph(val_dir, n_nodes, n_features, dt, n_timesteps, k, gnd_time_interval, noise_std=0)
    val_dataloader = DataloaderGraph(val_dataset, n_timesteps, shuffle=False)

    # --- plot the graph ---
    (X0, E0, _, _, _, _, _) = next(iter(train_dataloader))
    if show_plot == True:
        plotGraph(X0,E0)

    # --- model ---
    model = GNN(n_nodes, n_edges, n_features, n_latent, Y_features, gnd_time_interval)

    # --- loss function ---
    loss_fn = nn.MSELoss() # this is used for training the model

    # --- optimizer ---
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # --- scheduler ---
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    # --- training loop ---
    for i in range(epochs): # iterate over the epochs

        # train
        print("--- Epoch {0} ---".format(i+1))
        model.train(True)
        
        print("---- Train ----")
        for k, batch in tqdm(enumerate(train_dataloader)):

            # get next dataset
            (X, _, edge_attr, edge_index, target, _, _) = batch

            y_h, X_next = model(X, edge_index, edge_attr, dt, N=n_nodes) ## KEEP 

            # print("yh = ", y_h)
            # print("target = ", target)
            loss = loss_fn(y_h, target)
            n = int(k % n_timesteps)
            j = int((k - n) / n_timesteps)
            train_loss_list[i,j,n] = loss.item()

            # --- backpropagation ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # --- update the graph for the next time step
            X = X_next 
            X = X.detach_() # removes the tensor from the computational graph - it is now a leaf

        # log metrics to wandb
        wandb.log({"train_loss": np.mean(train_loss_list[i,:,:])}) # epochs, n_train files, n_timesteps

        # scheduler.step() # reduce the learning rate after every epoch

        # validate
        model.train(False)
        
        print("---- Validate ----")
        for k, batch in tqdm(enumerate(val_dataloader)):

            # get next dataset
            (X, _, edge_attr, edge_index, target, _, _) = batch

            y_h, X_next = model(X, edge_index, edge_attr, dt, N=n_nodes) 
    
            loss = loss_fn(y_h, target)
            n = int(k % n_timesteps) 
            j = int((k - n) / n_timesteps) 
            val_loss_list[i,j,n] = loss.item()

            # --- update the graph for the next time step
            X = X_next 
            X = X.detach_() # removes the tensor from the computational graph - it is now a leaf

        # log metrics to wandb
        wandb.log({"val_loss": np.mean(val_loss_list[i,:,:])}) # epochs, n_validation files, n_timesteps

        # save checkpoint 
        path = train_dir + "checkpoint_{0}.pt".format(i)
        if (i % checkpoint_period == 0):
            torch.save({
                "epoch" : i,
                "model_state_dict" : model.state_dict(),
                "optimizer_state_dict" : optimizer.state_dict(),
                "loss" : loss.item()
            }, path)

        # plot the loss curves for every nth epoch
        if (i % checkpoint_period == 0):
            plt.plot(list(range(n_timesteps)), train_loss_list[i,0,:], "-k", label="Train")
            plt.plot(list(range(n_timesteps)), val_loss_list[i,0,:], "-r", label="Validate")
            plt.xlabel("Timestep")
            plt.ylabel("MSE Loss")
            plt.title("Loss curve for 0th graph at {0}th epoch".format(i))
            plt.legend()
            filename = train_dir + "loss_curves_epoch_{0}.png".format(i)
            plt.savefig(filename)
            plt.clf()
            wandb.log({"nth epoch loss": wandb.Image(filename)})

    # --- compute mean loss for all time steps at each epoch ---
    train_loss_mean = np.mean(train_loss_list, axis=(1,2))    
    val_loss_mean = np.mean(val_loss_list, axis=(1,2))

    # plot the loss curves for all epochs
    plt.plot(list(range(epochs)), train_loss_mean[:], "-k", label="Train")
    plt.plot(list(range(epochs)), val_loss_mean[:], "-r", label="Validate")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.yscale("log")
    plt.title("Loss curve for all epochs")
    plt.legend()
    plt.grid(True)
    complete_loss_curve_filename = train_dir + "loss_curves_all_epochs.png"
    plt.savefig(complete_loss_curve_filename)
    plt.clf()
    wandb.log({"complete loss curve": wandb.Image(complete_loss_curve_filename)})

    print("Final train MSE loss = ", train_loss_mean[-1])
    # train_loss_pN, _ = sim2RealUnits(math.sqrt(train_loss_mean[-1]))
    # print("Final train loss [pn] = ", train_loss_pN)

    # log metrics to wandb
    # wandb.log({"final_train_MSE": train_loss_mean[-1], "final_train_loss_pn": train_loss_pN})
    wandb.log({"final_train_MSE": train_loss_mean[-1]})

    # --- save final checkpoint ---
    path = train_dir + "final_checkpoint.pt"
    if (i % checkpoint_period == 0):
        torch.save({
            "epoch" : i,
            "model_state_dict" : model.state_dict(),
            "optimizer_state_dict" : optimizer.state_dict(),
            "loss" : loss.item()
        }, path)

    # --- rollout a trajectory ---
    if show_rollout == True:
        print("---- Rollout ----")
        rollout_traj_file = train_dir + "rollout.dat"
        t0 = 100
        (X_norm, _, _, _, _, mean, std) = next(iter(train_dataloader))
        model.rollout(k, X_norm, mean, std, rollout_steps, rollout_traj_file, t0, top_file, traj_file, dt, n_nodes, n_features)      

    # [optional] finish the wandb run, necessary in notebooks
    wandb.finish()

if __name__ == "__main__":
    args = parse_arguments()

    # args.n_nodes=40
    # args.n_edges=20 
    # args.n_features=16 
    # args.n_latent=128 
    # args.Y_features=6 
    # args.dt=100 
    # args.tf=99900 
    # args.epochs=10 
    # args.lr=1e-4 
    # args.show_plot=True 
    # args.show_rollout=True 
    # args.rollout_steps=10 
    # args.top_file="/home/emma/repos/gnn-dna-sim/src/dataset-generation/dsDNA/top.top" 
    # args.traj_file="/home/emma/Documents/research/gnn-dna/dsdna-dataset/training/trajectory_sim_traj1.dat" 
    # args.train_dir="/home/emma/Documents/research/gnn-dna/dsdna-dataset/training/" 
    # args.val_dir="/home/emma/Documents/research/gnn-dna/dsdna-dataset/validation/" 
    # args.checkpoint_period=5 
    # args.n_train=8 
    # args.n_val=2 
    # args.seed=10707 
    

    main(args)
