from scipy.sparse import coo_matrix
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

def sim2RealUnits(force_sim=None, torque_sim=None):
    """
    Converts forces and torques in simulation units to real world units. 

    Inputs: 
    force_sim : scalar value of force in simulation units
    torque_sim : scalar value of torque in simulation units

    Outputs: 
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

def prepareEForModel(E):
        """
        Reduces edge attribute matrix to a sparser representation.

        Inputs: 
        E : edge attributes/adjacency matrix of shape [n_nodes, n_nodes]

        Outputs: 
        edge_attr : edge attributes in shape [n_edges, n_features_edges]
        edge_index : row and column indices for edges in COO format in matrix of shape [2, n_edges]

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

        return edge_attr, edge_index

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