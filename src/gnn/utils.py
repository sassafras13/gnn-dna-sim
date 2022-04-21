from scipy.sparse import coo_matrix
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

def doUpdate(X, Y, dt):
    """
    Updates the node attributes that describe each nucleotide's position (translational, rotational). Use a Euclidean update. 

    Each row in X contains the following entries: 
    0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15
    n   rx  ry  rz  bx  by  bz  nx  ny  nz  vx  vy  vz  Lx  Ly  Lz

    Each row in Y contains the following entries: 
    0   1   2   3   4   5
    ax  ay  az  atx aty atz

    Inputs:
    X : node attribute matrix containing each nucleotide's position and orientation in shape [n_nodes, n_features]
    Y : decoder output containing translational and rotational accelerations for each nucleotide in shape [n_nodes, n_state_vars]
    dt : scalar giving the time step of the ground truth data

    Outputs: 
    X_next : node attribute matrix for the next time step in shape [n_nodes, n_features]

    """
    # create X_next, the node attribute matrix for the next time step
    X_next = torch.zeros_like(X)

    # update translational velocity (vx, vy, vz)
    X_next[:, 10:13] = X[:, 10:13] + dt * Y[:, 0:3]

    # update rotational velocity (Lx, Ly, Lz)
    X_next[:, 13:] = X[:, 13:] + dt * Y[:, 3:]

    # update translational position (rx, ry, rz)
    X_next[:, 1:4] = X[:, 1:4] + dt * X_next[:, 10:13]

    # update backbone base versor (bx, by, bz)
    X_next[:, 4:7] = X[:, 4:7] + dt * X_next[:, 13:]

    # update normal versor (nx, ny, nz)
    X_next[:, 7:10] = X[:, 7:10] + dt * X_next[:, 13:]

    return X_next

def plotGraph(X, E):
    """
    Plots a 3D representation of the graph provided as input.

    Inputs:
    X : node attributes of shape [n_nodes, n_features]
    E : edge attributes/adjacency matrix of shape [n_nodes, n_nodes]

    Outputs:
    
    Plot of graph in 3D.
    """

    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.scatter3D(X[:,1], X[:,2], X[:,3], c=X[:,0], cmap="hsv") 

    for i in range(E.shape[0]):

        # get indices of nonzero elements
        idx = np.nonzero(E[i,:])
        idx = idx[0]

        for j in range(len(idx)):
            ax.plot3D([X[i,1], X[idx[j],1]], [X[i,2], X[idx[j],2]], [X[i,3], X[idx[j],3]], "k")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z") 
    ax.set_xlim(np.min(X[:,1:3]), np.max(X[:,1:3]))
    ax.set_ylim(np.min(X[:,1:3]), np.max(X[:,1:3]))
    ax.set_zlim(np.min(X[:,1:3]), np.max(X[:,1:3]))
    plt.show()


def makeGraphfromConfig(top_file, config_file, M=16):
    """
    Function builds graph from data contained in topology and configuration files for a structure.

    Inputs: 
    top_file : string containing full address of topology file (.top)
    config_file : string containing full address of configuration file (.oxdna)
    M : int indicating number of features, n_features, default 16.

    Outputs:
    X : node attributes of shape [n_nodes, n_features]
    E : edge attributes/adjacency matrix of shape [n_nodes, n_nodes]
    """
    # TODO: Update this to pull from the sim_out trajectory data instead of the configuration file!!!

    # trajectory
    # nucleotide dictionary
    nucleotide_dict = {
        "A": 0, 
        "T": 1,
        "C": 2, 
        "G": 3
    }

    # node attributes vector X
    # X.shape = N x M 
    # N = number of nodes
    # M = number of features describing each node

    # edge matrix 
    # E.shape = N X N 
    # 0 = no edge
    # 1 = nucleotides on same oligo
    # 2 = hydrogen bond

    with open(top_file) as f:
        lines = f.readlines()
        N = len(lines) - 1
        X = np.zeros((N,M))
        E = np.zeros((N,N))

        i = 0 
        count = 0 
        for line in lines:

            if count == 0:
                count += 1 
                continue

            # read through the 2nd thru last line of the .top file
            # X(i,0) = {0,1,2,3} for {A, T, C, G}
            for k in range(0,len(line)):
                if line[k] == "A" or line[k] == "C" or line[k] == "T" or line[k] == "G":
                    X[i,0] = nucleotide_dict[line[k]]
                    letter_idx = k
                    break

            # E(i,j) = 1 where j is the first, second numbers after the letter in the current row of .top file
            # if j = -1 do not add to E
            my_str = ""
            for k in range(letter_idx+2,len(line)):
                if line[k] != " ":
                    my_str += line[k]
                elif line[k] == " ":
                    j = int(my_str)
                    if j != -1:
                        E[i,j] = 1
                    my_str = ""
            
            # after loop ends add last number
            j = int(my_str)
            if j != -1:
                E[i,j] = 1
                
            i += 1

    with open(config_file) as f:
        lines = f.readlines()

        count = 0
        i = 0 
        for line in lines:
            if count < 3:
                count += 1
                continue

            if count >= 3:

                # X(i,1:-1) = all data in the current row of .oxdna file
                j = 1 
                my_str = ""
                for k in range(len(line)):
                    if line[k] != " " and line[k] != "\n":
                        my_str += line[k]
                    if line[k] == " " or line[k] == "\n":
                        X[i,j] = float(my_str)
                        j += 1
                        my_str = ""
                        
                i += 1

    return X, E

def makeGraphfromTraj(top_file, traj_file, M=16):
    """
    Function builds graph from data contained in topology and trajectory files for a structure.

    Inputs: 
    top_file : string containing full address of topology file (.top)
    traj_file : string containing full address of trajectory file (.dat)
    M : int indicating number of features, n_features, default 16.

    Outputs:
    X : node attributes of shape [n_nodes, n_features]
    E : edge attributes/adjacency matrix of shape [n_nodes, n_nodes]
    """

    # trajectory
    # nucleotide dictionary
    nucleotide_dict = {
        "A": 0, 
        "T": 1,
        "C": 2, 
        "G": 3
    }

    # node attributes vector X
    # X.shape = N x M 
    # N = number of nodes
    # M = number of features describing each node

    # edge matrix 
    # E.shape = N X N 
    # 0 = no edge
    # 1 = nucleotides on same oligo
    # 2 = hydrogen bond

    with open(top_file) as f:
        lines = f.readlines()
        N = len(lines) - 1
        X = np.zeros((N,M))
        E = np.zeros((N,N))

        i = 0 
        count = 0 
        for line in lines:

            if count == 0:
                count += 1 
                continue

            # read through the 2nd thru last line of the .top file
            # X(i,0) = {0,1,2,3} for {A, T, C, G}
            for k in range(0,len(line)):
                if line[k] == "A" or line[k] == "C" or line[k] == "T" or line[k] == "G":
                    X[i,0] = nucleotide_dict[line[k]]
                    letter_idx = k
                    break

            # E(i,j) = 1 where j is the first, second numbers after the letter in the current row of .top file
            # if j = -1 do not add to E
            my_str = ""
            for k in range(letter_idx+2,len(line)):
                if line[k] != " ":
                    my_str += line[k]
                elif line[k] == " ":
                    j = int(my_str)
                    if j != -1:
                        E[i,j] = 1
                    my_str = ""
            
            # after loop ends add last number
            j = int(my_str)
            if j != -1:
                E[i,j] = 1
                
            i += 1

    with open(traj_file) as f:
        lines = f.readlines()

        count = 0
        i = 0 
        for line in lines:
            if count < 3:
                count += 1
                continue

            if ((count >= 3) and (count < N)):

                # X(i,1:-1) = all data in the current row of .oxdna file
                j = 1 
                my_str = ""
                for k in range(len(line)):
                    if line[k] != " " and line[k] != "\n":
                        my_str += line[k]
                    if line[k] == " " or line[k] == "\n":
                        X[i,j] = float(my_str)
                        j += 1
                        my_str = ""
                        
                i += 1
                count += 1

    return torch.from_numpy(X).float(), torch.from_numpy(E).float()

def buildX(traj_file, t, X):
    """
    Builds the node attribute matrix for a given time step.

    Inputs: 
    traj_file : string indicating the location of the ground truth trajectory data
    t : scalar indicating current time step
    X : empty node attribute matrix in shape [n_nodes, n_features]

    Outputs: 
    X : node attribute matrix with values added from traj_file in shape [n_nodes, n_features]
    """
    # read through the file to find the current time step from "t = 100" etc.
    with open(traj_file) as f:
        lines = f.readlines()

        i = -1

        # find the line that contains the time step information
        for line in lines:
            i += 1
            if line[0:4] == "t = " and int(line[4:]) == t: 
                # print("found time {0} in trajectory data".format(t))

                count = 0 
                for line in lines[i:]:
                    # print("count", count)
                    # extract these lines and make a graph
                    if count < 3:
                        count += 1
                        continue

                    if ((count >= 3) and (count < (X.shape[0] + 3))):

                        # X(i,1:-1) = all data in the current row of .oxdna file
                        j = 1 
                        my_str = ""
                        for k in range(len(line)):
                            if line[k] != " " and line[k] != "\n":
                                my_str += line[k]
                            if line[k] == " " or line[k] == "\n":
                                X[(count-3),j] = float(my_str)
                                j += 1
                                my_str = ""
                        
                        count += 1
                    
                    else:
                        break
    
    return X

def getGroundTruthY(traj_file, t, dt, X, rand_idx):
    """
    Computes the ground truth acceleration for a given time step t for the nucleotides referenced in rand_idx. 

    Inputs: 
    traj_file : string indicating the location of the ground truth trajectory data
    t : scalar indicating current time step
    dt : scalar indicating size of time step
    X : node attribute matrix in shape [n_nodes, n_features]
    rand_idx: a Torch tensor containing randomly selected indices of nucleotides to compute accelerations for in shape [N, ]
    
    Ouputs:
    Y_target : a Torch tensor containing the ground truth accelerations for the randomly selected nucleotides in shape [N, n_state_vars]
    """
    # extract X_t and X_t+1 from the training data for those nucleotides and use them to compute the accelerations
    X_t = torch.zeros_like(X)
    X_t[:,0] = X[:,0]

    X_t1 = torch.zeros_like(X)
    X_t1[:,0] = X[:,0]

    # extract the data from that time step and the next one and build 2 X matrices, one for each time step
    X_t = buildX(traj_file, t, X_t)
    X_t1 = buildX(traj_file, t+dt, X_t1)
                                
    # find the v_t and v_t+1 from the indices in these graphs
    v_t = X_t[rand_idx, 10:]
    v_t1 = X_t1[rand_idx, 10:]

    # compute the target values 
    Y_target = (v_t1 - v_t) / dt 

    return Y_target