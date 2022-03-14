# Procedure for Generating Dataset

This README outlines how we generated a dataset containing multiple 2D and 3D DNA wireframe structures.

## Structure Design in Athena
We used Athena to generate .PDB files from several of the 2D and 3D structures in the tool's library. These .PDB files were converted to .top and .oxdna files using tacoxDNA, via the following command:

```
python3 PDB_oxdna.py <input filename> 53 
```

## Selecting oxDNA Parameters
We generated trajectories for each structure by running a 3-part simulation in oxDNA. First the structure was briefly **minimized**, then it was **relaxed** and finally we **simulated** the equilibriated structure. The min/relax/sim input files are contained in this subdirectory.

We use oxView to visualize the results of each step in the simulation process. We can save GIFs of the trajectories, and we select 3 pairs of nucleotides to track. Each pair spans a characteristic dimension of the structure - the goal is to understand when the structure equilibriates at a steady state configuration. The figure below shows how we select the base pairs in oxView. 

![Fig 1](https://github.com/sassafras13/gnn-dna-sim/blob/4d984f52a9ade5b7848ab336a1a0f8402204860b/src/dataset-generation/oxdna-bp-selection.png "Figure 1")     
Source: [1]  

We then feed the base pairs and a number of other files from the simulation run into oxDNA analysis tools to plot the distance between the pairs during the simulation below. We have a custom script that does this: ```runOxdna.py```, which can be called as:

```
python3 runOxdna.py <list of base pairs> <directory to oxdna-analysis-tools> <directory for output file> <oxDNA input file location> <trajectory file location>

python3 runOxdna.py /home/emma/Documents/Classes/10-707/final-project/wireframe-dataset/hexagon_base_pairs.csv /home/emma/repos/oxdna_analysis_tools/ /home/emma/Documents/Classes/10-707/final-project/wireframe-dataset/ /home/emma/Documents/Classes/10-707/final-project/wireframe-dataset/input_relax /home/emma/Documents/Classes/10-707/final-project/wireframe-dataset/trajectory_relax.dat
```

Next we compute the RMSF of the structure. 


Figure 2 shows the units used by the oxDNA simulation tool.

![Fig 2](https://github.com/sassafras13/gnn-dna-sim/blob/1c3de8192561159d4b1c9157ed3358d4f481899c/src/dataset-generation/oxdna-units.png "Figure 2")     
Source: [1]  
