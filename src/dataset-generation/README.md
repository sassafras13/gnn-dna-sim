# Procedure for Generating Dataset

This README outlines how we generated a dataset containing multiple 2D and 3D DNA wireframe structures.

## Structure Design in Athena
We used Athena to generate .PDB files from several of the 2D and 3D structures in the tool's library. These .PDB files were converted to .top and .oxdna files using tacoxDNA, via the following command:

```
python3 PDB_oxdna.py <input filename> 53 
```

## Selecting oxDNA Parameters
We generated trajectories for each structure by running a 3-part simulation in oxDNA. First the structure was briefly **minimized**, then it was **relaxed** and finally we **simulated** the equilibriated structure. The min/relax/sim input files are contained in this subdirectory.

Figure 1 shows the units used by the oxDNA simulation tool.

![Fig 1](https://github.com/sassafras13/gnn-dna-sim/blob/1c3de8192561159d4b1c9157ed3358d4f481899c/src/dataset-generation/oxdna-units.png "Figure 1")     
Source: [1]  
