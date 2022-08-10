# dir="/home/emma/Documents/Classes/10-707/final-project/wireframe-dataset/2D/hexagon"    
# dir="/home/emma/Documents/Classes/10-707/final-project/wireframe-dataset/3D/octahedron"
# dir="/home/emma/Documents/Classes/10-707/final-project/wireframe-dataset/3D/asymmetric-tetrahedron"
# dir="/home/emma/Documents/Classes/10-707/final-project/wireframe-dataset/3D/cube"
# dir="/home/emma/Documents/Classes/10-707/final-project/wireframe-dataset/3D/hexagonal-prism"
# dir="/home/emma/Documents/Classes/10-707/final-project/wireframe-dataset/3D/pentagonal-bipyramid"
# dir="/home/emma/Documents/Classes/10-707/final-project/wireframe-dataset/3D/pentagonal-pyramid"
# dir="/home/emma/Documents/Classes/10-707/final-project/wireframe-dataset/3D/tetrahedron"
# dir="/home/emma/Documents/Classes/10-707/final-project/wireframe-dataset/3D/triangular-bipyramid"
# dir="/home/emma/Documents/Classes/10-707/final-project/wireframe-dataset/2D/hexagon-mesh"
# dir="/home/emma/Documents/Classes/10-707/final-project/wireframe-dataset/2D/pentagon"
# dir="/home/emma/Documents/Classes/10-707/final-project/wireframe-dataset/2D/pentagon-mesh"
# dir="/home/emma/Documents/Classes/10-707/final-project/wireframe-dataset/2D/square"
# dir="/home/emma/Documents/Classes/10-707/final-project/wireframe-dataset/2D/triangle"
# dir="/home/emma/Documents/Classes/10-707/final-project/wireframe-dataset/2D/triangle-mesh"
dir="/home/emma/Documents/research/gnn-dna/dsdna-dataset/t2"

oxdna_tools_dir="/home/emma/repos/oxdna_analysis_tools"

# for the minimization, relaxation and simulation subfolders, run the runOxdnaAnalysis.py script

# # minimization
# min_dir="$dir/min_out"

# python3 runOxdnaAnalysis.py -b "$dir/base_pairs.csv" -x "$oxdna_tools_dir" -d "$min_dir/deviations.json" -o "$min_dir" -i "$min_dir/input_*" -t "$min_dir/trajectory_*.dat" -r "$min_dir/rmsf_min_plot.png" -y "$min_dir/rmsf_min_data.json" -e "$min_dir/energy_min.dat" 

# # relaxation
# relax_dir="$dir/relax_out"

# python3 runOxdnaAnalysis.py -b "$dir/base_pairs.csv" -x "$oxdna_tools_dir" -d "$relax_dir/deviations.json" -o "$relax_dir" -i "$relax_dir/input_*" -t "$relax_dir/trajectory_*.dat" -r "$relax_dir/rmsf_relax_plot.png" -y "$relax_dir/rmsf_relax_data.json" -e "$relax_dir/energy_relax.dat" 

# # simulation
# sim_dir="$dir/sim_out"

# python3 runOxdnaAnalysis.py -b "$dir/base_pairs.csv" -x "$oxdna_tools_dir" -d "$sim_dir/deviations.json" -o "$sim_dir" -i "$sim_dir/input_*" -t "$sim_dir/trajectory_*.dat" -r "$sim_dir/rmsf_sim_plot.png" -y "$sim_dir/rmsf_sim_data.json" -e "$sim_dir/energy_sim.dat" 

# minimization
echo "====== Min ======"
min_dir="$dir/min_out"
python3 runOxdnaAnalysis.py -x "$oxdna_tools_dir" -d "$min_dir/deviations.json" -o "$min_dir" -i "$min_dir/input_*" -t "$min_dir/trajectory_*.dat" -r "$min_dir/rmsf_min_plot.png" -y "$min_dir/rmsf_min_data.json" -e "$min_dir/energy_min.dat" 


# relaxation
echo "====== Relax ======"
relax_dir="$dir/relax_out"
python3 runOxdnaAnalysis.py -x "$oxdna_tools_dir" -d "$relax_dir/deviations.json" -o "$relax_dir" -i "$relax_dir/input_*" -t "$relax_dir/trajectory_*.dat" -r "$relax_dir/rmsf_relax_plot.png" -y "$relax_dir/rmsf_relax_data.json" -e "$relax_dir/energy_relax.dat" 

# simulation
echo "====== Sim ======"
sim_dir="$dir/sim_out"
python3 runOxdnaAnalysis.py -x "$oxdna_tools_dir" -d "$sim_dir/deviations.json" -o "$sim_dir" -i "$sim_dir/input_*" -t "$sim_dir/trajectory_*.dat" -r "$sim_dir/rmsf_sim_plot.png" -y "$sim_dir/rmsf_sim_data.json" -e "$sim_dir/energy_sim.dat" 

rm mean.dat
rm mean.json
