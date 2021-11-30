
# script to:
# write trajectory .dat file to .tfrecord
# compute average velocity and average acceleration for entire system
# return max bounding box dimensions
# save backbone and normal versors as step context?


import functools
import os
import json
import pickle

import tensorflow.compat.v1 as tf
from tqdm import tqdm
import numpy as np

from learning_to_simulate import reading_utils

############### READ IN THE DATA ###################
# Thanks Chris Kottke for code snippet

raw_data_path = "/tmp/datasets/Cuboid/trajectory_sim.dat"
framesAll = []

with open(raw_data_path, "r") as f:
    lines = f.readlines()
    frame = []
    for line in tqdm(lines):
        if "t =" in line:
            if len(frame) > 0:
                framesAll.append(frame)
                frame = []
        elif "b =" in line:
            pass
        elif "E =" in line:
            pass
        else:
            line_list = [float(s) for s in line.split()]
            frame.append(line_list[0:3])

if len(frame) > 0:
    framesAll.append(frame)

############### GENERATE POSITIONS, KEYS, PARTICLE TYPES ###################
positions = np.array([framesAll])
num_particles = len(positions[0][0])

keys = [0]

# generate a list of 7's that is as long as there are number of particles
particle_types = [7 * np.ones((num_particles,))]

print("positions length", len(positions))
print("num particles", num_particles)
print("positions[0][0]", positions[0][0])

############### GENERATE TFRECORD ###################


## Thanks: https://github.com/deepmind/deepmind-research/issues/199
# The following functions can be used to convert a value to a type compatible
# with tf.train.Example.

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# Write TF Record
with tf.python_io.TFRecordWriter("/tmp/datasets/Cuboid/train.tfrecord") as writer:
    
    for step, (particle_type, key, position) in enumerate(zip(particle_types, keys, positions)):
        print("particle type shape", particle_type.shape) # should be (n_particles,)
        print("position shape", position.shape) # should be (timesteps+1, n_particles, n_dims)
        seq = tf.train.SequenceExample(
                context=tf.train.Features(feature={
                    "particle_type": _bytes_feature(particle_type.tobytes()),
                    "key": _int64_feature(key)
                }),
                feature_lists=tf.train.FeatureLists(feature_list={
                    'position': tf.train.FeatureList(
                        feature=[_bytes_feature(position.flatten().tobytes())],
                    ),
                    'step_context': tf.train.FeatureList(
                        feature=[_bytes_feature(np.float32(step).tobytes())]
                    ),
                })
            )

        writer.write(seq.SerializeToString())


# this loads the data from a saved tf
# dt = tf.data.TFRecordDataset(['test.tfrecord'])
# dt = dt.map(functools.partial(reading_utils.parse_serialized_simulation_example, metadata=metadata))