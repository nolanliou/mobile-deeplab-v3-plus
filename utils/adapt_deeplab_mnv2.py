import urllib
import os
import sys

import tensorflow as tf

from deeplab_v3_plus import DeeplabV3Plus

flags = tf.app.flags

FLAGS = flags.FLAGS
flags.DEFINE_string('pretrained_model_dir', "", 'pretrained model dir.')
flags.DEFINE_string('output_dir', "output_dir", 'output dir')

def main(unused_argv):
    tf.reset_default_graph()

    # For simplicity we just decode jpeg inside tensorflow.
    # But one can provide any input obviously.
    file_input = tf.placeholder(tf.string, ())

    image = tf.image.decode_jpeg(tf.read_file(file_input))

    images = tf.expand_dims(image, 0)
    images = tf.cast(images, tf.float32) / 128. - 1
    images.set_shape((None, None, None, 3))
    images = tf.image.resize_images(images, (513, 513))
    model = DeeplabV3Plus(num_classes=21,
                          model_input_size=[513, 513],
                          output_stride=16,
                          weight_decay=0.01)

    logits = model.forward(
        images,
        '',
        is_training=False)

    exclude = ['global_step']
    variables_to_restore = tf.contrib.slim.get_variables_to_restore(
        exclude=exclude)
    variables_map = {}
    for v in variables_to_restore:
        old_name = v.name.split(':')[0]
        old_name = old_name.replace('conv2d/kernel', 'weights')
        old_name = old_name.replace('conv2d/bias', 'biases')
        variables_map[old_name] = v
        print(v)
    tf.train.init_from_checkpoint(FLAGS.pretrained_model_dir,
                                  variables_map)

    # download panda sample image
    filename, _ = urllib.urlretrieve(
        'https://upload.wikimedia.org/wikipedia/commons/f/fe/Giant_Panda_in_Beijing_Zoo_1.JPG')  # noqa

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        x = logits.eval(
            feed_dict={file_input: filename})
        if not os.path.exists(FLAGS.output_dir):
            os.makedirs(FLAGS.output_dir)
        saver.save(sess, FLAGS.output_dir + '/deeplab-v3-mnv2.ckpt')


if __name__ == '__main__':
    tf.app.run()
