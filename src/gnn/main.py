import argparse
import matplotlib.pyplot as plt
import numpy as np
from model import GNN, EdgeEncoder
import torch
from torch import nn
from torch_geometric.nn import MetaLayer
from torch_scatter import scatter_mean
from tqdm import tqdm

from utils import makeGraphfromTraj, plotGraph, getGroundTruthY, prepareEForModel, getForcesandTorques, sim2RealUnits, buildX

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Training GNN for DNA",
        fromfile_prefix_chars="@"
    )

    parser.add_argument("--n_features", type=int, default=16, help="Number of node features in input data")
    parser.add_argument("--n_latent", type=int, default=128, help="Number of latent features to encode node and edge attributes")
    parser.add_argument("--Y_features", type=int, default=6, help="Number of state variables output by model")
    parser.add_argument("--dt", type=int, default=100, help="Time interval in input data")
    parser.add_argument("--tf", type=int, default=50000, help="Final time step of input data")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--show_plot", type=bool, default=False, help="Show plot of initial structure")
    parser.add_argument("--show_rollout", type=bool, default=False, help="Generate a rollout trajectory")
    parser.add_argument("--rollout_steps", type=int, default=5, help="Number of steps in rollout trajectory")
    parser.add_argument("--dir", type=str, default="/home/emma/Documents/Classes/10-707/final-project/wireframe-dataset/2D/hexagon/", help="Location of input data")
    parser.add_argument("--checkpoint_period", type=int, default=5, help="Interval between saving checkpoints during training")
    parser.add_argument("--n_train", type=int, default=500, help="Number of training frames")
    parser.add_argument("--n_test", type=int, default=100, help="Number of test frames")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args, unknown = parser.parse_known_args()
    return args

def main(args):

    # --- define important parameters ---
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
    dir = args.dir
    checkpoint_period = args.checkpoint_period
    n_train = args.n_train
    n_test = args.n_test
    seed = args.seed

    # TODO: write a function that prints these conditions neatly at the start of the run

    # --- variables for storing loss ---
    train_loss_list = np.zeros((epochs, n_timesteps))
    # val_loss_list = np.zeros((epochs, n_test))

    Fmae_train_loss_list = np.zeros((epochs, n_timesteps))
    # Fmae_val_loss_list = np.zeros((epochs, n_test))    
    Tmae_train_loss_list = np.zeros((epochs, n_timesteps))
    # Tmae_val_loss_list = np.zeros((epochs, n_test))

    # --- select device to use ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    # --- input data ---

    # topology file
    # top_file = dir + "08_hexagon_DX_42bp-segid.pdb.top"
    top_file = dir + "top.top"

    # trajectory file
    # traj_file = dir + "sim_out/trajectory_sim.dat"
    traj_file = dir + "trajectory_sim.dat"

    # # ground truth file
    # gnd_truth_file = dir + "trajectory_sim.dat"

    # --- build the initial graph ---
    X_0, E_0 = makeGraphfromTraj(top_file, traj_file)
    n_nodes = X_0.shape[0]

    edge_attr_0, edge_index_0 = prepareEForModel(E_0)
    n_edges = edge_attr_0.shape[0]

    # --- build the graph for the start of the validation loop ---
    X_empty = torch.zeros_like(X_0)
    X_val_0 = buildX(traj_file, (n_train*dt), X_empty)

    # --- plot the graph ---
    if show_plot == True:
        plotGraph(X_0,E_0)

    # --- model ---
    model = GNN(n_nodes, n_edges, n_features, n_latent, Y_features)

    # --- loss function ---
    loss_fn = nn.MSELoss() # this is used for training the model
    mae_loss_fn = nn.L1Loss() # this is used for comparison with other models

    # --- optimizer ---
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # --- scheduler ---
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    # --- training loop ---
    for i in range(epochs): 

        X = X_0
        edge_attr, edge_index = edge_attr_0, edge_index_0

        # train
        print("--- Epoch {0} ---".format(i+1))
        model.train(True)
        train_t = t
        print("---- Train ----")
        for j in tqdm(range(n_timesteps)):        
        # for j in tqdm(range(n_train)):
            # if ((i % 1 == 0) and (j % 100 == 0)): 
            #     print("time ", j)
            #     print("X", X)
            #     print("edge_attr", edge_attr)

            rand_idx, preds, X_next = model(X, edge_index, edge_attr, dt, N=n_nodes) 

            target = getGroundTruthY(traj_file, train_t, dt, X_next, rand_idx)

            loss = loss_fn(preds, target)
            train_loss_list[i,j] = loss.item()
            # TODO: CALCULATE LOSS OVER ALL TIME STEPS
            # TODO: MAKE A NOTE ABOUT GNS BASELINE IS JUST X, Y, Z WITHOUT THETAS
            # TODO: emphasize that we are trying new things in the algorithms themselves, not just trying to learn on big structures
            # TODO: make a figure showing rollout and add to SI

            F_pred, T_pred, F_target, T_target = getForcesandTorques(preds, traj_file, n_nodes, train_t, dt)
            Fmae_loss = mae_loss_fn(F_pred, F_target)
            Fmae_train_loss_list[i,j] = Fmae_loss.item()
            Tmae_loss = mae_loss_fn(T_pred, T_target)
            Tmae_train_loss_list[i,j] = Tmae_loss.item()
            # print("Fmae train loss", Fmae_loss.item())
            # print("Tmae train loss", Tmae_loss.item())

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

        # # validate
        # model.train(False)
        # valid_t = train_t
        # X = X_val_0

        # print("---- Validate ----")
        # # for k in tqdm(range(n_timesteps)):
        # for k in tqdm(range(n_train, (n_train + n_test))):
        #     rand_idx, preds, X_next = model(X, edge_index, edge_attr, dt, N=100) 

        #     target = getGroundTruthY(traj_file, valid_t, dt, X_next, rand_idx)

        #     loss = loss_fn(preds, target)
        #     val_loss_list[i,(k-n_train)] = loss.item()

        #     F_pred, T_pred, F_target, T_target = getForcesandTorques(preds, gnd_truth_file, n_nodes, valid_t, dt)
        #     Fmae_loss = mae_loss_fn(F_pred, F_target)
        #     Fmae_val_loss_list[i,(k-n_train)] = Fmae_loss.item()
        #     Tmae_loss = mae_loss_fn(T_pred, T_target)
        #     Tmae_val_loss_list[i,(k-n_train)] = Tmae_loss.item()
        #     # print("validation loss", loss.item())
        #     # print("Fmae val loss", Fmae_loss.item())
        #     # print("Tmae val loss", Tmae_loss.item())

        #     # --- update the graph for the next time step
        #     X = X_next 
        #     X = X.detach_() # removes the tensor from the computational graph - it is now a leaf

        #     # update the time 
        #     valid_t += dt

        # save checkpoint 
        path = dir + "checkpoint_{0}.pt".format(i)
        if (i % checkpoint_period == 0):
            torch.save({
                "epoch" : i,
                "model_state_dict" : model.state_dict(),
                "optimizer_state_dict" : optimizer.state_dict(),
                "loss" : loss.item()
            }, path)

        # # plot the loss curves for every 10th epoch
        # if ((i+5) % 5 == 0):
        #     plt.plot(list(range(n_timesteps)), train_loss_list[i,:], "-k", label="Train")
        #     plt.plot(list(range(n_timesteps)), val_loss_list[i,:], "-r", label="Validate")
        #     plt.xlabel("Time step")
        #     plt.ylabel("Loss")
        #     plt.title("Loss curve for {0}th epoch".format(i))
        #     plt.legend()
        #     plt.savefig(dir + "loss_curves_epoch_{0}.png".format(i))
        #     plt.clf()

    # --- compute mean loss for all time steps at each epoch ---
    train_loss_mean = np.mean(train_loss_list, axis=1)
    # val_loss_mean = np.mean(val_loss_list, axis=1)
    Fmae_train_loss_mean = np.mean(Fmae_train_loss_list, axis=1)
    # Fmae_val_loss_mean = np.mean(Fmae_val_loss_list, axis=1)
    Tmae_train_loss_mean = np.mean(Tmae_train_loss_list, axis=1)
    # Tmae_val_loss_mean = np.mean(Tmae_val_loss_list, axis=1)

    # plot the loss curves for all epochs
    plt.plot(list(range(epochs)), train_loss_mean[:], "-k", label="Train")
    # plt.plot(list(range(epochs)), val_loss_mean[:], "-r", label="Validate")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.title("Loss curve for all epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(dir + "loss_curves_all_epochs.png")
    plt.clf()

    # plot the force MAE loss curves for all epochs
    plt.plot(list(range(epochs)), Fmae_train_loss_mean[:], "-k", label="Train")
    # plt.plot(list(range(epochs)), Fmae_val_loss_mean[:], "-r", label="Validate")
    plt.xlabel("Epoch")
    plt.ylabel("Force MAE Loss")
    plt.yscale("log")
    plt.title("Force MAE Loss curve for all epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(dir + "fmae_loss_curves_all_epochs.png")
    plt.clf()

    # plot the torque MAE loss curves for all epochs
    plt.plot(list(range(epochs)), Tmae_train_loss_mean[:], "-k", label="Train")
    # plt.plot(list(range(epochs)), Tmae_val_loss_mean[:], "-r", label="Validate")
    plt.xlabel("Epoch")
    plt.ylabel("Torque MAE Loss")
    plt.yscale("log")
    plt.title("Torque MAE Loss curve for all epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(dir + "tmae_loss_curves_all_epochs.png")

    print("Final train loss = ", train_loss_mean[-1])
    print("Final train force MAE loss = ", Fmae_train_loss_mean[-1])
    print("Final train torque MAE loss = ", Tmae_train_loss_mean[-1])

    train_loss_pN, _ = sim2RealUnits(train_loss_mean[-1])
    Fmae_train_loss_pN, _ = sim2RealUnits(Fmae_train_loss_mean[-1])
    _, Tmae_train_loss_pN = sim2RealUnits(None, Tmae_train_loss_mean[-1])

    print("Final train loss [pn] = ", train_loss_pN)
    print("Final train force MAE loss [pN] = ", Fmae_train_loss_pN)
    print("Final train torque MAE loss [pN nm] = ", Tmae_train_loss_pN)

    # --- save final checkpoint ---
    path = dir + "final_checkpoint.pt"
    if (i % checkpoint_period == 0):
        torch.save({
            "epoch" : i,
            "model_state_dict" : model.state_dict(),
            "optimizer_state_dict" : optimizer.state_dict(),
            "loss" : loss.item()
        }, path)

    # --- rollout a trajectory ---
    if show_rollout == True:
        rollout_traj_file = dir + "rollout.dat"
        t0 = 100
        model.rollout(rollout_steps, rollout_traj_file, t0, top_file, traj_file, dt, n_nodes)

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
