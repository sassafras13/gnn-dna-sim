
# script to:
# write trajectory .dat file to .tfrecord
# compute average velocity and average acceleration for entire system
# return max bounding box dimensions
# save backbone and normal versors as step context?

import argparse
import functools
import os
import json
import math
import pickle
import random
import tensorflow.compat.v1 as tf
from tqdm import tqdm
import numpy as np

from learning_to_simulate import reading_utils

############### TRAIN/VAL/TEST SPLIT ###################
def trainValTestSplit(n, per_train, per_val):
    # generate list of numbers n particles long
    all_idx = list(range(n))

    # shuffle
    random.shuffle(all_idx)

    # split into train/val/test sets
    train_idx = all_idx[0:math.ceil(per_train * n)]
    val_idx = all_idx[math.ceil(per_train * n): math.ceil(per_train * n) + math.ceil(per_val * n)]
    test_idx = all_idx[math.ceil(per_train * n) + math.ceil(per_val * n):]

    # return these sublists
    return train_idx, val_idx, test_idx

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

############### FUNCTION TO GENERATE POSITIONS, KEYS, PARTICLE TYPES AND SAVE ###################
def saveData(frames, filename):

    positions = [np.array(frames, dtype=np.float32)]
    num_particles = len(positions[0][0])

    particle_types = [7 * np.ones((num_particles,), dtype=np.int64)]
    keys = [np.int64(0)]


    print("positions length", len(positions))
    print("num particles", num_particles)
    # print("positions[0][0]", positions[0][0])

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

    # Write TF Record
    with tf.python_io.TFRecordWriter(filename) as writer:
        
        for step, (particle_type, key, position) in enumerate(zip(particle_types, keys, positions)):

            print("particle type shape", len(particle_type)) # should be (n_particles,)
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

def main():

    ############### ARGPARSE ###################
    parser = argparse.ArgumentParser(description="hi")
    parser.add_argument("--file_path", help="Path to file to convert to tfrecord", default="/tmp/datasets/Cuboid/trajectory_sim.dat")
    parser.add_argument("--train_split", help="Percentage of data to add to train file", default=0.8, type=float)
    parser.add_argument("--val_split", help="Percentage of data to add to validation file", default=0.1, type=float)
    parser.add_argument("--num_particles", help="Number of particles in file", default=10, type=int)
    args = parser.parse_args()

    ############### READ IN THE DATA ###################
    # Thanks Chris Kottke for code snippet

    raw_data_path = args.file_path
    framesAll = []
    trainFramesAll = []
    valFramesAll = []
    testFramesAll = []
    velocityAll = []

    train_idx, val_idx, test_idx = trainValTestSplit(args.num_particles, args.train_split, args.val_split)

    with open(raw_data_path, "r") as f:
        lines = f.readlines()
        frame = []
        velocity = []
        for line in tqdm(lines):
            if "t =" in line:
                if len(frame) > 0:
                    frameTrain = np.asarray(frame)[train_idx]
                    frameVal = np.asarray(frame)[val_idx]
                    frameTest = np.asarray(frame)[test_idx]

                    trainFramesAll.append(frameTrain)
                    valFramesAll.append(frameVal)
                    testFramesAll.append(frameTest)

                    framesAll.append(frame)
                    frame = []
                    velocityAll.append(velocity)
                    velocity = []


            elif "b =" in line:
                print(line)
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

        frameTrain = np.asarray(frame)[train_idx]
        frameVal = np.asarray(frame)[val_idx]
        frameTest = np.asarray(frame)[test_idx]
        
        trainFramesAll.append(frameTrain)
        valFramesAll.append(frameVal)
        testFramesAll.append(frameTest)

    positions = [np.array(framesAll, dtype=np.float32)]
    num_particles = len(positions[0][0])
    print("total number of particles", num_particles)

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

    saveData(trainFramesAll, "train.tfrecord")
    saveData(valFramesAll, "valid.tfrecord")
    saveData(testFramesAll, "test.tfrecord")

if __name__ == "__main__":
    main()
    