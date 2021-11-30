#import tensorflow as tf

#filename = "/tmp/datasets/WaterDropSample/train.tfrecord"
#filename = "/tmp/datasets/WaterDropSample/test.tfrecord"
#filename = "/tmp/datasets/Goop/train.tfrecord"

#for example in tf.python_io.tf_record_iterator(filename):
    #print(tf.train.SequenceExample.FromString(example))

# Import modules and this file should be outside learning_to_simulate code folder
import functools
import os
import json
import pickle

import tensorflow.compat.v1 as tf
import numpy as np

from learning_to_simulate import reading_utils
tf.enable_eager_execution()

# Set datapath and validation set
data_path = '/tmp/datasets/WaterDropSample'
filename = 'valid.tfrecord'

# Read metadata
def _read_metadata(data_path):
    with open(os.path.join(data_path, 'metadata.json'), 'rt') as fp:
        return json.loads(fp.read())

# Fetch metadata
metadata = _read_metadata(data_path)

print(metadata)

# Read TFRecord
ds_org = tf.data.TFRecordDataset([os.path.join(data_path, filename)])
ds = ds_org.map(functools.partial(reading_utils.parse_serialized_simulation_example, metadata=metadata))

# Convert to list
#@tf.function
def list_tf(ds):
    return(list(ds))
	
lds = list_tf(ds)

particle_types = []
keys = []
positions = []
for _ds in ds:
    context, features = _ds
    particle_types.append(context["particle_type"].numpy().astype(np.int64))
    keys.append(context["key"].numpy().astype(np.int64))
    positions.append(features["position"].numpy().astype(np.float32))
    print("context", context)
    print("features", features)
