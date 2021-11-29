import tensorflow as tf

#filename = "/tmp/datasets/WaterDropSample/train.tfrecord"
#filename = "/tmp/datasets/WaterDropSample/test.tfrecord"
filename = "/tmp/datasets/Goop/valid.tfrecord"

for example in tf.python_io.tf_record_iterator(filename):
    print(tf.train.Example.FromString(example))
