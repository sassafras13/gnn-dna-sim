# this script runs a complete analysis on the trajectory data for one DNA nanostructure

# import libraries
import argparse
import os
import numpy as np

# all functions required to read a configuration using the new RyeReader
from oxDNA_analysis_tools.UTILS.RyeReader import describe, get_confs, inbox

from analysis_tools import getEnergyPlot, getRMSF, getBondAnalysis, getPCA, importDataset, plotSampleTimeData, computePlotStatsPer, computePlotStatsAll, getAcclnData, plotCorrelation, computeStatsPer

# parse arguments
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Analyzing trajectory data from oxDNA",
        fromfile_prefix_chars="@"
    )

    parser.add_argument("--main_dir", type=str, help="Directory that contains dataset for one nanostructure and one min-relax-sim trajectory.")
    parser.add_argument("--topo_path", type=str, help="Path to topology file for nanostructure.")
    parser.add_argument("--input_path", type=str, help="Path to input file for sim part of trajectory generation.")    
    args, unknown = parser.parse_known_args()
    return args

def main(args):
    
    # === make subdirectories to store results of analysis ===
    main_subdir = os.path.join(args.main_dir, "analysis_results")
    oat_results_subdir = os.path.join(main_subdir, "oat_results")
    stats_results_subdir = os.path.join(main_subdir, "stats_results")
    os.makedirs(main_subdir, exist_ok=True)
    os.makedirs(oat_results_subdir, exist_ok=True)
    os.makedirs(stats_results_subdir, exist_ok=True)
    
    rmsd_subdir = os.path.join(oat_results_subdir, "rmsd")
    bonds_subdir = os.path.join(oat_results_subdir, "bonds")
    pca_subdir = os.path.join(oat_results_subdir, "pca")
    cluster_subdir = os.path.join(oat_results_subdir, "cluster")
    
    os.makedirs(os.path.join(oat_results_subdir, "energy"), exist_ok=True)
    os.makedirs(rmsd_subdir, exist_ok=True)
    os.makedirs(bonds_subdir, exist_ok=True)
    os.makedirs(pca_subdir, exist_ok=True)
    os.makedirs(cluster_subdir, exist_ok=True)
    
    position_subdir = os.path.join(stats_results_subdir, "position")
    base_subdir = os.path.join(stats_results_subdir, "base")
    base_normal_subdir = os.path.join(stats_results_subdir, "base-normal")
    velocity_subdir = os.path.join(stats_results_subdir, "velocity")
    ang_velocity_subdir = os.path.join(stats_results_subdir, "angular-velocity")
    accln_subdir = os.path.join(stats_results_subdir, "acceleration")
    ang_accln_subdir = os.path.join(stats_results_subdir, "angular-acceleration")
    
    os.makedirs(position_subdir, exist_ok=True)
    os.makedirs(base_subdir, exist_ok=True)
    os.makedirs(base_normal_subdir, exist_ok=True)
    os.makedirs(velocity_subdir, exist_ok=True)
    os.makedirs(ang_velocity_subdir, exist_ok=True)
    os.makedirs(accln_subdir, exist_ok=True)
    os.makedirs(ang_accln_subdir, exist_ok=True)
    # os.makedirs(os.path.join(stats_results_subdir, "energy"), exist_ok=True)
    # os.makedirs(os.path.join(stats_results_subdir, "correlation-plots"), exist_ok=True)
    

    # === run OAT tools analysis ===
    
    # get trajectory and topology information objects
    traj = os.path.join(args.main_dir, "sim_out/trajectory_sim.dat")
    top_info, traj_info = describe(args.topo_path, traj)

    # # energy plot
    # getEnergyPlot(args.main_dir)
    
    # RMSD plot and RMSF JSON file
    getRMSF(traj_info, top_info, rmsd_subdir)
    
    # bond analysis JSON file
    getBondAnalysis(args.main_dir, args.input_path, bonds_subdir)
    
    # PCA JSON file
    # cluster JSON and .dat files
    getPCA(args.main_dir, pca_subdir, cluster_subdir)    

    # === run stats analysis ===
    
    # import data
    posAll, baseAll, baseNormalAll, velocityAll, angVelocityAll, bbAll, energyAll = importDataset(args.main_dir)
    
    # convert to Numpy arrays
    posAll = np.array(posAll)
    baseAll = np.array(baseAll)
    baseNormalAll = np.array(baseNormalAll)
    velocityAll = np.array(velocityAll)
    angVelocityAll = np.array(angVelocityAll)
    
    # get acceleration data
    acclnAll, angAcclnAll = getAcclnData(velocityAll, angVelocityAll)
    
    # plot sample time data
    # position data, base and base normal vectors, velocity, angular velocity, accln
    plotSampleTimeData(posAll, "Position", os.path.join(position_subdir, "position_time_data.png"))
    plotSampleTimeData(baseAll, "Base Vector", os.path.join(base_subdir, "base_time_data.png"))
    plotSampleTimeData(baseNormalAll, "Base Normal Vector", os.path.join(base_normal_subdir, "base_normal_time_data.png"))
    plotSampleTimeData(velocityAll, "Velocity", os.path.join(velocity_subdir, "velocity_time_data.png"))
    plotSampleTimeData(angVelocityAll, "Angular Velocity", os.path.join(ang_velocity_subdir, "ang_velocity_time_data.png"))
    plotSampleTimeData(acclnAll, "Acceleration", os.path.join(accln_subdir, "accln_time_data.png"))
    plotSampleTimeData(angAcclnAll, "Angular Acceleration", os.path.join(ang_accln_subdir, "ang_accln_time_data.png"))

    # mean, std dev, min, max per nucleotide
    computePlotStatsPer(posAll, "Position", os.path.join(position_subdir, "position_per_nucleotide.png"))
    computePlotStatsPer(baseAll, "Base Vector", os.path.join(base_subdir, "base_per_nucleotide.png"))
    computePlotStatsPer(baseNormalAll, "Base Normal Vector", os.path.join(base_normal_subdir, "base_normal_per_nucleotide.png"))
    computePlotStatsPer(velocityAll, "Velocity", os.path.join(velocity_subdir, "velocity_per_nucleotide.png"))
    computePlotStatsPer(angVelocityAll, "Angular Velocity", os.path.join(ang_velocity_subdir, "ang_velocity_per_nucleotide.png"))
    computePlotStatsPer(acclnAll, "Acceleration", os.path.join(accln_subdir, "accln_per_nucleotide.png"))
    computePlotStatsPer(angAcclnAll, "Angular Acceleration", os.path.join(ang_accln_subdir, "ang_accln_per_nucleotide.png"))
    
    # mean, std dev, min, max per timestep
    computePlotStatsPer(posAll, "Position", os.path.join(position_subdir, "position_per_timestep.png"), axis=1)
    computePlotStatsPer(baseAll, "Base Vector", os.path.join(base_subdir, "base_per_timestep.png"), axis=1)
    computePlotStatsPer(baseNormalAll, "Base Normal Vector", os.path.join(base_normal_subdir, "base_normal_per_timestep.png"), axis=1)
    computePlotStatsPer(velocityAll, "Velocity", os.path.join(velocity_subdir, "velocity_per_timestep.png"), axis=1)
    computePlotStatsPer(angVelocityAll, "Angular Velocity", os.path.join(ang_velocity_subdir, "ang_velocity_per_timestep.png"), axis=1)
    computePlotStatsPer(acclnAll, "Acceleration", os.path.join(accln_subdir, "accln_per_timestep.png"), axis=1)
    computePlotStatsPer(angAcclnAll, "Angular Acceleration", os.path.join(ang_accln_subdir, "ang_accln_per_timestep.png"), axis=1)
    
    # mean, std dev, min, max over all
    computePlotStatsAll(posAll, "Position", os.path.join(position_subdir, "position_mean.png"))
    computePlotStatsAll(baseAll, "Base Vector", os.path.join(base_subdir, "base_mean.png"))
    computePlotStatsAll(baseNormalAll, "Base Normal Vector", os.path.join(base_normal_subdir, "base_normal_mean.png"))
    computePlotStatsAll(velocityAll, "Velocity", os.path.join(velocity_subdir, "velocity_mean.png"))
    computePlotStatsAll(angVelocityAll, "Angular Velocity", os.path.join(ang_velocity_subdir, "ang_velocity_mean.png"))
    computePlotStatsAll(acclnAll, "Acceleration", os.path.join(accln_subdir, "accln_mean.png"))
    computePlotStatsAll(angAcclnAll, "Angular Acceleration", os.path.join(ang_accln_subdir, "ang_accln_mean.png"))
    
    # correlation plots between position, base, base normal 
    plotCorrelation(posAll, baseAll, 0, "Mean Position", "Mean Base Vector", os.path.join(position_subdir, "position_vs_base_per_nucleotide_corr.png"))
    plotCorrelation(posAll, baseNormalAll, 0, "Mean Position", "Mean Base Normal Vector", os.path.join(position_subdir, "position_vs_base_normal_per_nucleotide_corr.png"))
    plotCorrelation(posAll, baseAll, 1, "Mean Position", "Mean Base Vector", os.path.join(position_subdir, "position_vs_base_per_timestep_corr"))
    plotCorrelation(posAll, baseNormalAll, 1, "Mean Position", "Mean Base Normal Vector", os.path.join(position_subdir, "position_vs_base_normal_per_timestep_corr"))
    
    # correlation plots between velocity, angular velocity
    plotCorrelation(velocityAll, angVelocityAll, 0, "Mean Velocity", "Mean Angular Velocity", os.path.join(velocity_subdir, "velocity_vs_ang_velocity_per_nucleotide_corr.png"))
    plotCorrelation(velocityAll, angVelocityAll, 1, "Mean Velocity", "Mean Angular Velocity", os.path.join(velocity_subdir, "velocity_vs_ang_velocity_per_timestep_corr"))

    # energy data
    # mean, std dev, min, max over all time steps
    _energyAll = np.array(energyAll)
    energyAll = _energyAll[:, 0] # E = Etot U K -- we just want Etot
    mean_E, std_E, min_E, max_E = computeStatsPer(energyAll)
    print("mean energy = {0:.3f}".format(mean_E))
    print("std dev energy = {0:.3f}".format(std_E))
    print("min energy = {0:.3f}".format(min_E))
    print("max energy = {0:.3f}".format(max_E))
    
    
if __name__ == "__main__":
    args = parse_arguments()
    main(args)




