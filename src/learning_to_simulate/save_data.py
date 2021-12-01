
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
velocityAll = []

with open(raw_data_path, "r") as f:
    lines = f.readlines()
    frame = []
    velocity = []
    for line in tqdm(lines):
        if "t =" in line:
            if len(frame) > 0:
                framesAll.append(frame)
                frame = []
                velocityAll.append(velocity)
                velocity = []
        elif "b =" in line:
            # print(line)
            pass
        elif "E =" in line:
            pass
        else:
            line_list = [float(s) for s in line.split()]
            frame.append(line_list[0:3])
            velocity.append(line_list[-6:-3])

if len(frame) > 0:
    framesAll.append(frame)
    velocityAll.append(velocity)

############### COMPUTE AVG VELOCITY AND ACCLN ###################
dt = 0.005
velocityAll = np.array(velocityAll)
print("velocity shape", velocityAll.shape)

meanVelocity = np.mean(velocityAll, axis=0)
meanVelocity = np.mean(meanVelocity, axis=0)
print("mean velocity", meanVelocity)

stdVelocity = np.std(velocityAll, axis=0)
stdVelocity = np.std(stdVelocity, axis=0)
print("std velocity", stdVelocity)

acclnAll = []
for i in range(velocityAll.shape[0]):
    accln = []
    for j in range(velocityAll.shape[1]-1):
        v_curr = velocityAll[i][j]
        v_next = velocityAll[i][j+1]
        a = (v_next - v_curr) / dt
        accln.append(a)
    acclnAll.append(accln)

meanAccln = np.mean(acclnAll, axis=0)
meanAccln = np.mean(meanAccln, axis=0)
print("mean accln", meanAccln)

stdAccln = np.std(acclnAll, axis=0)
stdAccln = np.std(stdAccln, axis=0)
print("std accln", stdAccln)

############### GENERATE POSITIONS, KEYS, PARTICLE TYPES ###################
positions = [np.array(framesAll, dtype=np.float32)]
# print("positions shape", positions[0].shape)
# print("\n")
num_particles = len(positions[0][0])

particle_types = [7 * np.ones((num_particles,), dtype=np.int64)]
keys = [np.int64(0)]


# print("positions length", len(positions))
# print("num particles", num_particles)
# print("positions[0][0]", positions[0][0])

############### GENERATE TFRECORD ###################
# print("type of particle types", type(particle_types))
# print("type of keys", type(keys))
# print("type of positions", type(positions))
# print("\n")
# print("type of particle types0", type(particle_types[0]))
# print("type of keys0", type(keys[0]))
# print("type of positions0", type(positions[0]))
# print("\n")
# print("type of particle types single entry",type(particle_types[0][0]))
# print("type of positions single entry", type(positions[0][0][0][0]))

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

        # print("particle type shape", len(particle_type)) # should be (n_particles,)
        # print("position shape", position.shape) # should be (timesteps+1, n_particles, n_dims)

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