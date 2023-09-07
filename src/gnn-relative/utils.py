from scipy.sparse import coo_matrix
import torch
from torch import nn, Tensor
import math
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple

def reverseNormalizeX(X_norm : Tensor, mean : Tensor, std : Tensor):
    """
    Reverses the normalization of node attribute matrix X.
    """
    X = (X_norm * std) + mean
    return X

def normalizeX(X : Tensor, sum : Tensor, total_n : int, sos : Tensor):
    """
    Normalizes the node attribute matrix X and updates running statistics.

    Parameters:
    -----------
    X : Tensor
        Node attribute matrix for one time step of size [n_nodes, n_features]
    sum : Tensor
        Sum of all nodes for all time steps for each column of size [1, n_features]
    total_n : int
        Total number of nodes considered for all time steps so far
    sos : Tensor
        Sum of squares (x_i - mean)**2 for all time steps for all nodes for each column of size [1, n_features]

    Returns:
    --------
    X : Tensor
        Normalized version of X
    sum : Tensor
        Updated version of sum
    total_n : int
        Updated version of total_n
    sos : Tensor
        Updated version of sos
    mean : Tensor
        Mean computed for all nodes so far for each column of size [1, n_features]
    std : Tensor
        Standard deviation for all nodes so far for each column of size [1, n_features]
    """

    # update sum of entries in every column, use all nodes
    sum += torch.sum(X, 0) # sum over all the columns

    # update total number of nodes, n -- count each node individually
    total_n += X.shape[0]

    # mean = sum / n  
    mean = sum / total_n

    # update sum of squares in every column, use all nodes
    sos += torch.sum(((X-mean)**2),0)

    # recompute mean and variance
    # sigma = sum (x_i - mean_x)^2 / (n-1)
    sigma = sos / (total_n - 1)

    # std = sqrt(sigma)
    std = torch.sqrt(sigma)

    # apply normalization
    # for each entry in each column, x_i - mean / std
    X = (X - mean) / std

    return X, sum, total_n, sos, mean, std

def getKNN(X : Tensor, k : int = 3):
    """
    Computes the adjacency matrix E for the node attribute matrix X by finding the k nearest neighbors to each node.

    Parameters: 
    -----------
    X : Tensor
        Node attribute matrix containing only position data for every node of size [n_nodes, 2] or [n_nodes, 3] depending on 2-D or 3-D coordinates.
    k : int
        The number of nearest neighbors to be used to construct adjacency matrix.

    Returns:
    --------
    E : Tensor
        The adjacency matrix of size [X.shape[0], X.shape[0]]
    """
    # create a matrix for holding the pairwise distances between all points of size [X.shape[0], X.shape[0]]
    distances = torch.zeros([X.shape[0], X.shape[0]])  

    # iterate through all the nodes in X 
    for i in range(X.shape[0]):
        # then iterate through all the nodes ahead of current node
        for j in range(i, X.shape[0]):
            if i == j:
                distances[i,j] = float("inf")
            else:
                # compute the Euclidean distance between this pair and add to distances matrix
                distances[i,j] = math.dist(X[i,:],X[j,:])
                distances[j,i] = distances[i,j]

    # create a new matrix E of size [X.shape[0], X.shape[0]]
    E = torch.zeros([X.shape[0], X.shape[0]])  

    # find the k smallest distances for each row of the distance matrix and get their column numbers
    _, indices = torch.topk(distances, k=k, dim=0, largest=False, sorted=False)

    # add 1 to every (i,j) and (j,i) entry of E using indices found above
    rows = torch.arange(0,X.shape[0])
    for i in range(indices.shape[0]):
        E[rows,indices[i,:]] = 1

    # return E
    return E

def sim2RealUnits(force_sim=None, torque_sim=None):
    """
    Converts forces and torques in simulation units to real world units. 

    Parameters: 
    -----------
    force_sim : scalar value of force in simulation units
    torque_sim : scalar value of torque in simulation units

    Returns:
    --------
    force_real_pN : scalar value of force in [pN]
    torque_real_pNnm : scalar value of torque in [pN nm]
    """
    if force_sim is not None:
        # force_sim units are (5.24 * 10^-25) kg * (8.518 * 10^-10) m / (3.03 * 10^-12 s)^2
        # want to convert to kg * m / s^2 = N
        # then convert to pN = 10^-12 N
        
        # force real = force sim * (5.24 * 10^-25) kg * (8.518 * 10^-10) m * 1 / ((3.03 * 10^-12)s)^2 
        # force_real = force_sim * (1 / (5.24 * 10**(-25))) * ((1 / (8.518 * 10**(-10)))) * ((3.03 * 10**(-12))**2)
        force_real = force_sim * (5.24 * 10**(-25)) * (8.518 * 10**(-10)) * (1 / ((3.03 * 10**(-12))**2))

        # force_real pN = force real * (1 pN / 10^-12 N)
        force_real_pN = force_real * (1 / (10**(-12)))
    else:
        force_real_pN = None

    if torque_sim is not None: 
        # torque_sim units are (5.24 * 10^-25) kg * (8.518 * 10^-10)^2 m / (3.03 * 10^-12 s)^2
        # want to convert to kg * m^2 / s^2 = Nm
        # then convert to pN nM = 10^-12 N 10^-9 m

        # torque real = torque sim * (5.24 * 10^-25) kg * ((8.518 * 10^-10) m)^2) * 1 / ((3.03 * 10^-12)s)^2 
        torque_real = torque_sim * (5.24 * 10**(-25)) * ((8.518 * 10**(-10))**2) * (1 / ((3.03 * 10**(-12))**2))

        # torque_real pN = torque real * (1 pN / 10^-12 N) * (1 nm / 10^-9 m)
        torque_real_pNnm = torque_real * (1 / (10**(-12))) * (1 / (10**(-9)))
    else: 
        torque_real_pNnm = None

    return force_real_pN, torque_real_pNnm


def doUpdate(X, Y, dt, gnd_time_interval):
    """
    Updates the node attributes that describe each nucleotide's position (translational, rotational). Use a Euclidean update. 

    Each row in X contains the following entries: 
    0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15
    n   rx  ry  rz  bx  by  bz  nx  ny  nz  vx  vy  vz  Lx  Ly  Lz

    Each row in Y contains the following entries: 
    0   1   2   3   4   5
    ax  ay  az  atx aty atz

    Parameters:
    -----------
    X : Tensor
        node attribute matrix containing each nucleotide's position and orientation in shape [n_nodes, n_features]
    Y : Tensor
        decoder output containing translational and rotational accelerations for each nucleotide in shape [n_nodes, n_state_vars]
    dt : int
        scalar giving the time step of the ground truth data
    gnd_time_interval : float
        Time represented by one time step in the ground truth data

    Returns: 
    --------
    X_next : Tensor
        node attribute matrix for the next time step in shape [n_nodes, n_features]

    """

    delta_t = dt * gnd_time_interval

    # create X_next, the node attribute matrix for the next time step
    X_next = torch.zeros_like(X)

    # update translational velocity (vx, vy, vz)
    X_next[:, 10:13] = X[:, 10:13] + delta_t * Y[:, 0:3]

    # update rotational velocity (Lx, Ly, Lz)
    X_next[:, 13:] = X[:, 13:] + delta_t * Y[:, 3:]

    # update translational position (rx, ry, rz)
    X_next[:, 1:4] = X[:, 1:4] + delta_t * X_next[:, 10:13]

    # update backbone base versor (bx, by, bz)
    X_next[:, 4:7] = X[:, 4:7] + delta_t * X_next[:, 13:]

    # update normal versor (nx, ny, nz)
    X_next[:, 7:10] = X[:, 7:10] + delta_t * X_next[:, 13:]

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

        for j in range(len(idx)):
            ax.plot3D([X[i,1], X[idx[j],1]], [X[i,2], X[idx[j],2]], [X[i,3], X[idx[j],3]], "k")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z") 
    # ax.set_xlim(np.amin(X[:,1:3]), np.amax(X[:,1:3]))
    # ax.set_ylim(np.amin(X[:,1:3]), np.amax(X[:,1:3]))
    # ax.set_zlim(np.amin(X[:,1:3]), np.amax(X[:,1:3]))
    plt.show()


def makeGraphfromConfig(top_file, config_file, M=16):
    """
    Function builds graph from data contained in topology and configuration files for a structure.

    Parameters: 
    -----------
    top_file : str
        string containing full address of topology file (.top)
    config_file : str
        string containing full address of configuration file (.oxdna)
    M : int 
        integer indicating number of features, n_features, default 16.

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

def makeGraphfromTraj(top_file: str, 
                      traj_file: str, 
                      n_nodes: int, 
                      n_features:int=16)->Tuple[Tensor, Tensor]:
    """
    Function builds graph from data contained in topology and trajectory files for a structure.
    Specifically data from the first time step is used to build the graph.

    Parameters: 
    -----------
    top_file : str
        full address of topology file (.top)
    traj_file : str
        full address of trajectory file (.dat)
    n_nodes : int
        number of nodes
    n_features : int
        number of features, default 16.

    Returns:
    --------
    X : Tensor
        node attributes of shape [n_nodes, n_features] 
        X = [nucleotide #, position, backbone-base versor, normal versor, velocity, angular velocity]
        all dynamics data is for the first time step in the given trajectory .dat file
    E : Tensor
        edge attributes/adjacency matrix of shape [n_nodes, n_nodes]
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
        X = np.zeros((n_nodes,n_features))
        E = np.zeros((n_nodes,n_nodes))

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

            if ((count >= 3) and (count < (N+3))):

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

def buildX(traj_file, n_timesteps, dt, n_nodes, n_features)->Tensor:
    """
    Builds the node attribute matrix for all time steps.

    Parameters: 
    -----------
    traj_file : str
        string indicating the location of the ground truth trajectory data
    n_timesteps : int
        scalar indicating total number of timesteps in file
    dt : int
        size of time step (in simulation units)
    n_nodes : int
        number of nodes in graph
    n_features : int
        number of features for each node

    Returns: 
    --------
    full_X : Tensor
        node attribute matrix with values added from traj_file in shape [n_timesteps+1, n_nodes, n_features]
    """
    # nucleotide dictionary
    nucleotide_dict = {
        "A": 0, 
        "T": 1,
        "C": 2, 
        "G": 3
    }
    X = np.zeros((n_timesteps+1, n_nodes, n_features))

    # read through the file to find the current time step from "t = 100" etc.
    with open(traj_file) as f:
        lines = f.readlines()

        line_number = -1 # this keeps track of the current line

        # iterate through all the lines in the file
        for line in lines:
            line_number += 1

            # if the line contains a time stamp, then we update the current time index
            if line[0:4] == "t = ": 
                t_index = int(int(line[4:]) / dt)-1

                count = 0 
                for line in lines[line_number:]:

                    # extract these lines and make a graph
                    # skip the first three rows containing time, bounding box, energy information
                    if count < 3:
                        count += 1
                        continue

                    # add the remaining trajectory data to X
                    if ((count >= 3) and (count < (n_nodes + 3))):

                        # X(line_number,1:-1) = all data in the current row of .oxdna file
                        j = 1 
                        my_str = ""
                        for k in range(len(line)):
                            if line[k] != " " and line[k] != "\n":
                                my_str += line[k]
                            if line[k] == " " or line[k] == "\n":
                                X[t_index,(count-3),j] = float(my_str)
                                j += 1
                                my_str = ""
                        
                        count += 1
                    
                    else:
                        break
    
    return torch.from_numpy(X).float()

def getGroundTruthY(traj_file: str,
                    j: int,
                    full_X : Tensor, 
                    dt: int, 
                    n_nodes: int,
                    n_features: int,
                    gnd_time_interval: float)->Tensor:
    """
    Computes the ground truth acceleration for a given time step t. 

    Parameters:
    ----------- 
    traj_file : str
        string indicating the location of the ground truth trajectory data
    j : int
        scalar indicating current time step (index for a tensor, not in simulation units)
    full_X : Tensor
        a Torch Tensor containing the trajectory data for all nodes for all time steps of shape [n_timesteps, n_nodes, n_features]
    dt : int
        size of time step (in simulation units)
    n_nodes : int
        number of nodes in graph
    n_features : int
        number of features for each node
    gnd_time_interval : float
        Time represented by one time step in the ground truth data
    
    Returns:
    --------
    Y_target : Tensor
        a Torch tensor containing the ground truth accelerations
    """
    delta_t = dt * gnd_time_interval

    # extract the data from that time step and the previous one and build 2 X matrices, one for each time step
    X_t = full_X[0:n_nodes]
    X_t1 = full_X[n_nodes:]
                                
    # find the v_t and v_t+1 from the indices in these graphs
    v_t = X_t[:, 10:]
    v_t1 = X_t1[:, 10:]

    # compute the target values 
    Y_target = (v_t1 - v_t) / delta_t
    
    return Y_target

def prepareEForModel(E: Tensor)->Tuple[Tensor, Tensor]:
    """
    Reduces edge attribute matrix to a sparser representation.

    Parameters: 
    -----------
    E : Tensor
        edge attributes/adjacency matrix of shape [n_nodes, n_nodes]

    Returns: 
    --------
    edge_attr : Tensor
        edge attributes in shape [n_edges, n_features_edges]
    edge_index : Tensor 
        row and column indices for edges in COO format in matrix of shape [2, n_edges]
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

    return edge_attr, edge_index, edge_index_coo

def getForcesandTorques(Y, gnd_truth_file, n_nodes, t, dt):
    """
    This function takes in the model predicted accelerations and the location of the file containing the ground truth forces and torques, 
    and returns the predicted forces and torques as well as the ground truth values for use in computing the loss. 

    Each row in Y contains the following entries: 
    0   1   2   3   4   5
    ax  ay  az  atx aty atz

    Inputs: 
    Y : shape [n_nodes, Y_features]

    Outputs: 
    Fx_pred, Fy_pred, Fz_pred : 
    T_pred : 
    F_target : 
    T_target : 
    """

    # TODO: Clean up these comments following Eric's formatting approach

    # take in the model predicted vector Y and extract translational and rotational accelerations
    mean_Y = torch.mean(Y,1)

    # compute predicted force as F_pred = m * a 
    # m is the mass of one nucleotide
    # ssDNA has a molecular weight of 303.7g/mol
    # 1 mol contains 6.02 x 10^23 nucleotides
    # 1 nucleotide = 303.7g / mol * 1 mol / 6.02 * 10^23 nucleotides = 5.05 * 10^-22 g / nt
    # a is the average acceleration of one particle
    # everything is in simulation units
    # 5.05 * 10^-22 g = 5.05 * 10^-25 kg ~= 1 unit of simulation mass (5.24 * 10^-25 kg)
    # so F_pred ~= a
    Fx_pred = mean_Y[0]
    Fy_pred = mean_Y[1]
    Fz_pred = mean_Y[2]
    F_pred = torch.mean(torch.tensor([[Fx_pred, Fy_pred, Fz_pred]]))

    # compute predicted torque as T_pred = I * rotational a
    # I is the moment of inertia (we approximate as a sphere)
    # I = 2/5 * m * r^2 
    # I ~= 0.4 * 1 * (0.6nm * 1 length unit / 0.8518nm)
    # I ~= 0.28
    # rotational a is the average rotational acceleration of one nucleotide
    # T_pred = 0.28 * rotational a
    Tx_pred = 0.28 * mean_Y[3]
    Ty_pred = 0.28 * mean_Y[4]
    Tz_pred = 0.28 * mean_Y[5]
    T_pred = torch.mean(torch.tensor([[Tx_pred, Ty_pred, Tz_pred]]))

    # we go into the trajectory file 
    # look for the correct time (listed as step = ...)
    # extract the rows of data
    # return the last 6 cols (fx, fy, fz, tx, ty, tz)
    # read through the file to find the current time step from "t = 100" etc.
    X_gnd_truth = torch.zeros((n_nodes, 15))
    count = 0 # need to count to 3 then can start extracting data
    i = -1
    with open(gnd_truth_file) as f:
        lines = f.readlines()

        for line in lines:
            i += 1
            # find the line that contains t - dt timestep 
            # i.e. if we want time point 200, find time point 100 because the time point 200 data is below it
            if (line[0:7] == "step = ") and (int(line[7:12]) == t) and (count == 0):
            
                for line in lines[i:]:

                    # extract these lines and make a graph
                    if count <= 3:
                        count += 1
                        continue

                    if ((count > 3) and (count < (n_nodes + 4))):
                        # X(i,1:-1) = all data in the current row of .oxdna file
                        j = 0 
                        my_str = ""
                        for k in range(len(line)-1):
                            if line[k] != " " and line[k] != "\n":
                                my_str += line[k]
                            if line[k] == " " or line[k] == "\n":
                                X_gnd_truth[(count-4),j] = float(my_str)
                                j += 1
                                my_str = ""
                            
                        count += 1
                    
                    else:
                        break
        mean_X = torch.mean(X_gnd_truth, 0)
        Fx_target = mean_X[9]
        Fy_target = mean_X[10]
        Fz_target = mean_X[11]
        F_target = torch.mean(torch.tensor([[Fx_target, Fy_target, Fz_target]]))

        Tx_target = mean_X[12]
        Ty_target = mean_X[13]
        Tz_target = mean_X[14]
        T_target = torch.mean(torch.tensor([[Tx_target, Ty_target, Tz_target]]))

        return F_pred, T_pred, F_target, T_target