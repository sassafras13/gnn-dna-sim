#!/bin/sh

# run the relaxation procedure

cp /home/mmbl/Documents/ebenjami/tigges-design/tigges_design_staples.json.top top.top
cp /home/mmbl/Documents/ebenjami/tigges-design/tigges_design_staples.json.oxdna conf.conf

mkdir min_out
mkdir relax_out
mkdir sim_out

echo "start input_min----------------------------------------------------------------------------------------------------------------------"
oxDNA input_min

echo "start input_relax--------------------------------------------------------------------------------------------------------------------"
oxDNA input_relax

echo "start input_sim-----------------------------------------------------------------------------------------------------------------------"
oxDNA input_sim

rm top.top
rm conf.conf
