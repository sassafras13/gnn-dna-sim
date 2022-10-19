import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from oxDNA_analysis_tools.bond_analysis import bond_analysis
from oxDNA_analysis_tools.mean import mean
from oxDNA_analysis_tools.deviations import deviations
from oxDNA_analysis_tools.deviations import output
from oxDNA_analysis_tools.pca import pca

def getEnergyPlot(main_dir, time_oxdna=3.03, energy_oxdna=4.142):
    """
    Generates a plot of the total energy over time for the data in the sim_out subdirectory of main_dir. 
    
    Inputs:
    
    main_dir: str, the path to the directory containing the min-relax-sim trajectory data for a DNA nanostructure. 
    time_oxdna: float, 3.03ps = 1 unit of time in oxDNA
    energy_oxdna: float, 4.142e-20J is one unit of energy in oxDNA
    """
    
    # read in first and last columns
    energy_file = os.path.join(main_dir, "sim_out/energy_sim.dat")
    output_dir = os.path.join(main_dir, "analysis_results/oat_results/energy/")
    data = np.genfromtxt(energy_file, delimiter='  ')

    # convert first column to time
    dt = 0.005 # assuming this will be true for all simulations
    time = data[:,0] * dt * time_oxdna 

    # convert last column to energy
    # for MD sims the energy file has the following format:
    # [time (steps * dt)]	[potential energy]	[kinetic energy]	[total energy]
    total_energy = data[:,3] * energy_oxdna 

    # plot data
    plt.figure()
    plt.plot(time, total_energy)
    plt.xlabel("Time [ps]")
    plt.ylabel("Total Energy [e-20 J]")
    plt.title("Energy for {}".format(main_dir))
    plt.grid()
    plt.savefig("{}/energy.png".format(output_dir))
    plt.close()
    
def getRMSF(traj_info, top_info, output_dir, ncpus=4):
    """
    Computes both root-mean-square fluctuation (RMSF) and root-mean-square deviations (RMSD) for a trajectory.
    """
    
    # Compute the mean structure and RMSFs
    mean_conf = mean(traj_info, top_info, ncpus=4)
    RMSDs, RMSFs = deviations(traj_info, top_info, mean_conf, ncpus=4)
    
    # save the results
    outfile = os.path.join(output_dir, "devs.json")
    rmsd_plot_name = os.path.join(output_dir, "rmsd.png")
    rmsd_data_file = os.path.join(output_dir, "rmsd_op.json")
    output(RMSDs, RMSFs, outfile, rmsd_plot_name, rmsd_data_file)
    
def getBondAnalysis(main_dir, input_file, output_dir, ncpus=4):
    """
    Calculates bond occupancy for a given trajectory.
    
    bond_analysis (-p <n_cpus>) <input> <trajectory> <designed_pairs_file> <output_file>
    """
    
    trajectory_file = os.path.join(main_dir, "sim_out/trajectory_sim.dat")
    pairs_file = os.path.join(main_dir, "pairs.txt")
    output_file = os.path.join(output_dir, "bonds.json")
    
    os.system("oat bond_analysis -p {0} {1} {2} {3} {4}".format(ncpus, input_file, trajectory_file, pairs_file, output_file))
    
def getPCA(main_dir, output_dir, cluster_subdir, ncpus=4):
    """
    Performs principal component analysis over deviations per nucleotide during trajectory. Note that we also perform clustering in case the user wants to find clusters later. We need to move the files generated by the clustering step to the right directory after they are generated.
    
    Requires computing mean configuration before performing analysis. 
    
    Mean:
    oat mean [-h] [-p num_cpus] [-o output_file] [-d deviation_file]
                [-i index_file] [-a alignment_configuration]
                trajectory
    
    PCA:
    oat pca [-h] [-p num_cpus] [-c] [-n num_components]
               trajectory meanfile outfile
    """
    
    mean_file = os.path.join(main_dir, "sim_out/mean.dat")
    trajectory_file = os.path.join(main_dir, "sim_out/trajectory_sim.dat")
    output_file = os.path.join(output_dir, "pca.json")
    
    os.system("oat mean -p {0} -o {1} {2}".format(ncpus, mean_file, trajectory_file))
    
    os.system("oat pca -p {0} -c {1} {2} {3}".format(ncpus, trajectory_file, mean_file, output_file))
    
    os.system("mv scree.png coordinates2.png cluster_data.json cluster_*.dat centroid_*.dat animated.mp4 {0}".format(cluster_subdir))
    return None
   
def importDataset(main_dir):
    """
    Imports configuration data as lists. Thanks Chris Kottke for code snippet.
    """

    
    raw_data_path = os.path.join(main_dir, "sim_out/trajectory_sim.dat")

    posAll = []
    baseAll = []
    baseNormalAll = []
    velocityAll = []
    angVelocityAll = []
    bbAll = [] # bounding box data
    energyAll = [] 

    with open(raw_data_path, "r") as f:
        lines = f.readlines()
        pos = []
        base = []
        baseNormal = []
        velocity = []
        angVelocity = []

        for line in tqdm(lines):
            if "t =" in line:
                if len(pos) > 0:
                    # save and reset the frames
                    posAll.append(pos)
                    pos = []
                    baseAll.append(base)
                    base = []
                    baseNormalAll.append(baseNormal)
                    baseNormal = []
                    velocityAll.append(velocity)
                    velocity = []
                    angVelocityAll.append(angVelocity)
                    angVelocity = []

            elif "b =" in line:
                line_list = [float(s) for s in line.split()[2:]]
                bbAll.append(line_list)

            elif "E =" in line:
                line_list = [float(s) for s in line.split()[2:]]
                energyAll.append(line_list)

            else:
                line_list = [float(s) for s in line.split()]
                pos.append(line_list[0:3])
                base.append(line_list[3:6])
                baseNormal.append(line_list[6:9])
                velocity.append(line_list[-6:-3])
                angVelocity.append(line_list[-3:])

    if len(pos) > 0:
        posAll.append(pos)
        baseAll.append(base)
        baseNormalAll.append(baseNormal)
        velocityAll.append(velocity)
        angVelocityAll.append(angVelocity)
    
    return posAll, baseAll, baseNormalAll, velocityAll, angVelocityAll, bbAll, energyAll

def plotSampleTimeData(data, description, filename):
    """
    Plots some sample trajectories over time.
    
    Inputs:
    
    data : np array size [t, n, c] where t = time steps, n = number of nucleotides, c = number of axes
    description : str describing the data
    """
    t = np.arange(0, data.shape[0])
    for i in range(0, data.shape[1], 8):
        plt.plot(t, data[:,i,0], label="{0}".format(i))
    plt.xlabel("Time")
    plt.ylabel("X {0}".format(description))
    plt.title("Sample {0} Data".format(description))
    plt.grid()
    plt.legend()
    plt.savefig(filename)
    plt.close()
    
def computeStatsPer(data, axis=0):
    """
    Calculates mean, standard deviation, min, max for each nucleotide over all time steps for a given dataset.
    
    Inputs: 
    
    data : np array size [t, n, c] where t = time steps, n = number of nucleotides, c = number of axes
    axis : int, the axis along which we want to compute the mean (axis = 0 will average over time, axis = 1 over nucleotides)
    """
    mean_per = np.mean(data, axis=axis)
    std_per = np.std(data, axis=axis)
    min_per = np.min(data, axis=axis)
    max_per = np.max(data, axis=axis)
    
    return mean_per, std_per, min_per, max_per

def computeStatsAll(mean_per, std_per, min_per, max_per):
    """
    Calculates mean, standard deviation, min, max for all nucleotides over all time steps for a given dataset.
    
    Inputs: 
    
    mean_per : np array size [n or t, c] where t = time steps, n = number of nucleotides, c = number of axes, contains mean data
    std_per : np array size [n or t, c] where t = time steps, n = number of nucleotides, c = number of axes, contains std data
    min_per : np array size [n or t, c] where t = time steps, n = number of nucleotides, c = number of axes, contains min data
    max_per : np array size [n or t, c] where t = time steps, n = number of nucleotides, c = number of axes, contains max data
    """

    mean_all = np.mean(mean_per, axis=0)
    std_all = np.std(std_per, axis=0)
    min_all = np.min(min_per, axis=0)
    max_all = np.max(max_per, axis=0)
    
    return mean_all, std_all, min_all, max_all

def computePlotStatsPer(data, description, filename, axis=0):
    """
    Calculates and plots mean, standard deviation, min, max for each nucleotide over all time steps for a given dataset.
    
    Inputs: 
    
    data : np array size [t, n, c] where t = time steps, n = number of nucleotides, c = number of axes
    description : str describing the data
    axis : int, the axis along which we want to compute the mean (axis = 0 will average over time, axis = 1 over nucleotides)
    """
    
    mean_per, std_per, min_per, max_per = computeStatsPer(data, axis)
    
    x = np.arange(0,mean_per.shape[0])
    
    # mean data
    plt.errorbar(x, mean_per[:,0], yerr=std_per[:,0], fmt="or", label="x")
    plt.errorbar(x, mean_per[:,1], yerr=std_per[:,1], fmt="og", label="y")
    plt.errorbar(x, mean_per[:,2], yerr=std_per[:,2], fmt="ob", label="z")
    
    # min max data
    plt.plot(x, min_per[:,0], "--r", label="min x")
    plt.plot(x, min_per[:,1], "--g", label="min y")
    plt.plot(x, min_per[:,2], "--b", label="min z")
    plt.plot(x, max_per[:,0], "-r", label="max x")
    plt.plot(x, max_per[:,1], "-g", label="max y")
    plt.plot(x, max_per[:,2], "-b", label="max z")

    if axis == 0:
        plt.xlabel("Nucleotide")
        plt.title("Mean {0} per Nucleotide".format(description))
    elif axis == 1:
        plt.xlabel("Time")
        plt.title("Mean {0} per Time Step".format(description))
        
    plt.ylabel("Mean {0}".format(description))
    plt.legend()
    plt.grid()
    plt.savefig(filename)
    plt.close()
    
def computePlotStatsAll(data, description, filename):
    """
    Calculates and plots mean, standard deviation, min, max for each nucleotide over all time steps for a given dataset.
    
    Inputs: 
    
    data : np array size [t, n, c] where t = time steps, n = number of nucleotides, c = number of axes
    description : str describing the data
    """
        
    mean_per, std_per, min_per, max_per = computeStatsPer(data, axis=0)
    mean_all, std_all, min_all, max_all = computeStatsAll(mean_per, std_per, min_per, max_per)
    
    x = ("x", "y", "z")
    plt.errorbar(x, mean_all, yerr=std_all, fmt="ob", label="mean")
    plt.plot(x, min_all, label="min")
    plt.plot(x, max_all, label="max")
    plt.xlabel("Nucleotide")
    plt.ylabel("Mean {0}".format(description))
    plt.title("Mean {0} for all Nucleotides".format(description))
    plt.grid()
    plt.legend()
    plt.savefig(filename)
    plt.close()
    
def plotGradientScatter(mean_d1, mean_d2, coord, data1label, data2label, filename):
    """
    Plot the mean of data 1 and data 2 over one coordinate (x, y, z) using a gradient to plot the points to indicate their evolution over time.
    
    Inputs:
    mean_d1 : np array size [t, c] where t = time steps, c = number of axes
    mean_d2 : np array size [t, c] where t = time steps, c = number of axes
    coord : int, a value in [0, 2] which indicates whether we should plot x, y or z coordinate
    data1label: str, the name of data1
    data2label: str, the name of data2
    """
    
    if coord == 0:
        coord_label = "x"
    elif coord == 1:
        coord_label = "y"
    else:
        coord_label = "z"

    plt.scatter(mean_d1[:,coord], mean_d2[:,coord], c=np.arange(mean_d1.shape[0]), cmap="plasma")
    plt.grid()
    plt.xlabel("{0}".format(data1label))
    plt.ylabel("{0}".format(data2label))
    plt.title("Correlation over Time: {0} and {1} in {2}".format(data1label, data2label, coord_label))
    plt.savefig(filename + "_" + coord_label + ".png")
    plt.close()

def plotCorrelation(data1, data2, axis, data1label, data2label, filename):
    """
    Compute the mean of data1 and data2 and plot against each other on a scatter plot to look for correlations.
    
    Inputs:
    data1 : np array size [t, n, c] where t = time steps, n = number of nucleotides, c = number of axes
    data2 : np array size [t, n, c] where t = time steps, n = number of nucleotides, c = number of axes
    axis : int, the axis along which we want to compute the mean (axis = 0 will average over time, axis = 1 over nucleotides)
    data1label: str, the name of data1
    data2label: str, the name of data2
    """
    # compute mean data for all particles
    mean_d1, _, _, _ = computeStatsPer(data1, axis=axis)
    mean_d2, _, _, _ = computeStatsPer(data2, axis=axis)
    
    if axis == 0:

        # plot means on scatter plot for x, y, z coordinates
        plt.scatter(mean_d1[:,0], mean_d2[:,0], label="x")
        plt.scatter(mean_d1[:,1], mean_d2[:,1], label="y")
        plt.scatter(mean_d1[:,2], mean_d2[:,2], label="z")
        plt.legend()
        plt.grid()
        plt.xlabel("{0}".format(data1label))
        plt.ylabel("{0}".format(data2label))
        plt.title("Correlation over Nucleotides: {0} and {1}".format(data1label, data2label))
        plt.savefig(filename)
        plt.close()
        
    if axis == 1: 
        
        for i in range(data1.shape[2]):
            plotGradientScatter(mean_d1, mean_d2, i, data1label, data2label, filename)
            
def getAcclnData(velocityAll, angVelocityAll, dt=0.005):
    # stats over accln
    acclnAll = []
    angAcclnAll = []

    # iterate over nucleotides
    # velocityAll = [t, n, c]
    for i in range(velocityAll.shape[1]):
        accln = []
        angAccln = []

        # iterate over time steps
        for j in range(velocityAll.shape[0]-1):

            v_curr = velocityAll[j, i, :]
            v_next = velocityAll[j+1, i, :]
            a = (v_next - v_curr) / dt
            accln.append(a)

            av_curr = angVelocityAll[j, i, :]
            av_next = angVelocityAll[j+1, i, :]
            av = (av_next - av_curr) / dt
            angAccln.append(av)

        acclnAll.append(accln)
        angAcclnAll.append(angAccln)
        
    _acclnAll = np.array(acclnAll)
    acclnAll = np.reshape(_acclnAll, (_acclnAll.shape[1], _acclnAll.shape[0], _acclnAll.shape[2]))
    _angAcclnAll = np.array(angAcclnAll)
    angAcclnAll = np.reshape(_angAcclnAll, (_angAcclnAll.shape[1], _angAcclnAll.shape[0], _angAcclnAll.shape[2]))
    
    return acclnAll, angAcclnAll