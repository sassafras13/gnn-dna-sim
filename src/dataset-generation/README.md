# Procedure for Generating Dataset

This README outlines how we generated a dataset containing multiple 2D and 3D DNA wireframe structures.

## Structure Design in Athena
We used Athena to generate .PDB files from several of the 2D and 3D structures in the tool's library. These .PDB files were converted to .top and .oxdna files using tacoxDNA, via the following command:

```
python3 PDB_oxdna.py <input filename> 53 
```

## Selecting oxDNA Parameters
We generated trajectories for each structure by running a 3-part simulation in oxDNA. First the structure was briefly **minimized**, then it was **relaxed** and finally we **simulated** the equilibriated structure. The min/relax/sim input files are contained in this subdirectory. This is a [useful description](https://dna.physics.ox.ac.uk/index.php/Documentation) of all the parameters in the input files. We **save the output to terminal** to record the running time of the simulations for comparison with other baseline tests later.

We use oxView to visualize the results of each step in the simulation process. We can save GIFs of the trajectories, and we select 3 pairs of nucleotides to track. Each pair spans a characteristic dimension of the structure - the goal is to understand when the structure equilibriates at a steady state configuration. The figure below shows how we select the base pairs in oxView. 

![Fig 1](https://github.com/sassafras13/gnn-dna-sim/blob/4d984f52a9ade5b7848ab336a1a0f8402204860b/src/dataset-generation/oxdna-bp-selection.png "Figure 1")     
Source: [1]  

We then feed the base pairs and a number of other files from the simulation run into a custom script that calls [oxDNA analysis tools](https://github.com/sulcgroup/oxdna_analysis_tools) to plot the distance between the pairs during the simulation below. This script also calculates the root-mean-square fluctuations (RMSF) for the input trajectory, and saves the results as a JSON file that can be overlaid on the topology + trajectory (.dat) files in oxView. The custom script  ```runOxdna.py``` can be called as:

```
python3 runOxdna.py -b <list of base pairs> -x <directory to oxdna-analysis-tools> -d <directory to save deviations JSON> -o <directory for output file> -i <oxDNA input file location> -t <trajectory file location>

python3 runOxdna.py -b /home/emma/Documents/Classes/10-707/final-project/wireframe-dataset/hexagon_base_pairs.csv -x /home/emma/repos/oxdna_analysis_tools -d /home/emma/Documents/Classes/10-707/final-project/wireframe-dataset/hexagon_deviations.json -o /home/emma/Documents/Classes/10-707/final-project/wireframe-dataset/ -i /home/emma/Documents/Classes/10-707/final-project/wireframe-dataset/input_sim -t /home/emma/Documents/Classes/10-707/final-project/wireframe-dataset/trajectory_sim.dat
```

TODO: Add a bash script that does this all for a given shape.

Figure 2 shows the units used by the oxDNA simulation tool.

![Fig 2](https://github.com/sassafras13/gnn-dna-sim/blob/1c3de8192561159d4b1c9157ed3358d4f481899c/src/dataset-generation/oxdna-units.png "Figure 2")     
Source: [1]  


