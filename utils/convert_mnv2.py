# This script must be ran at the root directory of 'tensorflow/models'
import sys
sys.path.append('research/slim')

import tensorflow as tf
from nets.mobilenet import mobilenet_v2

tf.reset_default_graph()

# For simplicity we just decode jpeg inside tensorflow.
# But one can provide any input obviously.
file_input = tf.placeholder(tf.string, ())

image = tf.image.decode_jpeg(tf.read_file(file_input))

images = tf.expand_dims(image, 0)
images = tf.cast(images, tf.float32) / 128.  - 1
images.set_shape((None, None, None, 3))
images = tf.image.resize_images(images, (224, 224))

# Note: arg_scope is optional for inference.
with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=False)):
    logits, endpoints = mobilenet_v2.mobilenet(images)
        
# Restore using exponential moving average since it produces (1.5-2%) higher 
# accuracy
#ema = tf.train.ExponentialMovingAverage(0.999)
#vars = ema.variables_to_restore()
saver = tf.train.Saver()

from datasets import imagenet

checkpoint_name = 'mobilenet_v2_1.0_224'
checkpoint='checkpoint/' + checkpoint_name + '.ckpt'

with tf.Session() as sess:
    saver.restore(sess,  checkpoint)
    x = endpoints['Predictions'].eval(feed_dict={file_input: 'panda.jpg'})
    saver.save(sess, 'hehe/mobilenet_v2_1.0_224.ckpt')
label_map = imagenet.create_readable_names_for_imagenet_labels()  
print("Top 1 prediction: ", x.argmax(),label_map[x.argmax()], x.max())
