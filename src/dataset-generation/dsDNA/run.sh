#!/bin/sh

mkdir min_out
mkdir relax_out
mkdir sim_out

echo "start input_min----------------------------------------------------------------------------------------------------------------------"
/home/emma/repos/oxDNA/bin/oxDNA input_min

read -p "Paused to copy info. Press [Enter] key to continue..." myarg

echo "start input_relax--------------------------------------------------------------------------------------------------------------------"
/home/emma/repos/oxDNA/bin/oxDNA input_relax

read -p "Paused to copy info. Press [Enter] key to continue..." myarg

echo "start input_sim-----------------------------------------------------------------------------------------------------------------------"
/home/emma/repos/oxDNA/bin/oxDNA input_sim

