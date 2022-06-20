# This script runs oxdna analysis on simulation data to extract length data as a function of sim time.
# The data is then exported to be analyzed in a Jupyter notebook.
# Many thanks to Erik Poppleton, Joakim Bohlin, Michael Matthies, Shuchi Sharma, Fei Zhang, Petr Sulc: Design, optimization, and analysis of large DNA and RNA nanostructures through interactive visualization, editing, and molecular simulation. (2020) Nucleic Acids Research e72. https://doi.org/10.1093/nar/gkaa417

# import libraries
import os
import csv
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# function that calls oxdna analysis tool in command line
def runDistanceAnalysis(base_pairs, oxdna_dir, output_dir, input_file, traj_file):

    # build command to run oxdna analysis tools
    # the format for calling distance.py is:
    # (-c -o <output> -f <histogram/trajectory/both> -d <data file output>)...
    # -i <<input> <trajectory> <particleID 1> <particleID 2> (<particleID 1> <particleID 2> ...)>
   
    command = 'python3 {0}/src/oxDNA_analysis_tools/distance.py -o output -d {1}/traj_data.txt -i {2} {3} {4}'.format(oxdna_dir, output_dir, input_file, traj_file, base_pairs) 

    # call the distance.py function in oxdna analysis tools
    os.system(command)

def runRMSFAnalysis(oxdna_dir, deviation_file, traj_file):

    # build command to run oxdna analysis tools 
    # the format for calling compute_mean.py is: 
    # -p <n_cpus> -f <oxDNA/json/both> -o <mean structure> -d <deviations file> -i <index file> -a <align conf id>

    command = "python3 {0}/src/oxDNA_analysis_tools/compute_mean.py -f both -d {1} {2}".format(oxdna_dir, deviation_file, traj_file)
    os.system(command)

def getRMSFPlot(oxdna_dir, RMSF_plot, traj_file, RMSF_file):
    mean_structure = "mean.json"
    command = "python3 {0}/src/oxDNA_analysis_tools/compute_deviations.py -o {1} -r {2} {3} {4}".format(oxdna_dir, RMSF_file, RMSF_plot, mean_structure, traj_file)
    os.system(command)

# function that builds string of base pairs to be studied
def buildBasePairStr(base_pair_file):

    # read the base pair CSV file
    with open(base_pair_file) as f:
        reader = csv.reader(f)
        
        # skip first row
        #next(reader)

        # create empty list 
        base_pairs_list = [] 

        for row in reader:

            # append each pair of bases to a string
            base_pairs_list.append(str(row[0]))
            base_pairs_list.append(str(row[1]))

        base_pairs = ' '.join(base_pairs_list)

    return base_pairs

def plotTrajectoryDataPLT(output_dir):
    data = output_dir + "/traj_data.txt"
    data_array = np.loadtxt(data, skiprows=1)
    T = data_array.shape[0]-1

    plt.figure()
    for i in range(data_array.shape[1]):
        plt.plot(list(range(T)), data_array[1:,i], label="Pair {0}".format(i))
    plt.xlabel("Simulation Time")
    plt.ylabel("Distance Between Bases")
    plt.legend()
    # plt.show()
    plt.savefig("{}/trajectories.png".format(output_dir))

def plotEnergyData(energy_file, output_dir):

    # read in first and last columns
    data = np.genfromtxt(energy_file,
                     delimiter='  ')

    # convert first column to time
    dt = 0.005 # assuming this will be true for all simulations
    time = data[:,0] * dt * 3.03 # 3.03 ps = 1 unit of time in oxdna

    # convert last column to energy
    total_energy = data[:,3] * 4.142 # e-20 J

    # plot data
    plt.figure()
    plt.plot(time, total_energy)
    plt.xlabel("Time [ps]")
    plt.ylabel("Energy [e-20 J]")
    plt.savefig("{}/energy.png".format(output_dir))

# wrapper function 
def doAnalysis(base_pair_file, oxdna_dir, deviation_file, output_dir, input_file, traj_file, RMSF_plot, RMSF_file, energy_file):

    # # call buildBasePairList to get list of base pairs
    # base_pairs = buildBasePairStr(base_pair_file)

    # # call runOxdnaAnalysis with base_pairs to generate analysis file as a text file
    # runDistanceAnalysis(base_pairs, oxdna_dir, output_dir, input_file, traj_file)

    # # plot the resulting data using Altair and save
    # plotTrajectoryDataPLT(output_dir)

    print("====== RMSF Analysis ======")
    runRMSFAnalysis(oxdna_dir, deviation_file, traj_file)

    print("====== RMSF Plot ======")
    getRMSFPlot(oxdna_dir, RMSF_plot, traj_file, RMSF_file)

    print("====== Energy Analysis ======")
    plotEnergyData(energy_file, output_dir)

# main
if __name__ == "__main__":

    # the arguments to be passed into this script are:
    # - the directory where the list of base pairs are
    # - the directory where the oxdna analysis tools are located
    # - the directory where the output data should be saved
    # - the path to the simulation input file
    # - the path to the simulation trajectory file
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", '--base_pairs', help='Enter the path to the CSV file containing the list of base pairs to be studied.', type=str)
    parser.add_argument("-x", '--oxdna_dir', help='Enter the directory where the oxdna-analysis-tools repo is located.', type=str)
    parser.add_argument("-d", "--deviation_file", help="Enter the name to save the deviations file to.", type=str)
    parser.add_argument("-o", '--output_dir', help='Enter the directory where the output data file should be stored.', type=str)
    parser.add_argument("-i", '--input_file', help='Enter the directory where the oxdna simulation input file is stored.', type=str)
    parser.add_argument("-t", '--traj_file', help='Enter the directory where the trajectory file generated during simulation is stored.', type=str)
    parser.add_argument("-r", "--RMSF_plot", help="Enter the filename for saving the RMSF plot.", type=str)
    parser.add_argument("-y", "--RMSF_file", help="Enter the filename for saving the RMSF data.", type=str)
    parser.add_argument("-e", "--energy_file", help="Enter the directory where the energy file is stored.", type=str)
    args = parser.parse_args()

    # call doAnalysis
    doAnalysis(args.base_pairs, args.oxdna_dir, args.deviation_file, args.output_dir, args.input_file, args.traj_file, args.RMSF_plot, args.RMSF_file, args.energy_file)
