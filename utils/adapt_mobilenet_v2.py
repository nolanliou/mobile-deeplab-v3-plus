import urllib
import os

import tensorflow as tf

from mobilenet_v2 import MobilenetV2

flags = tf.app.flags

FLAGS = flags.FLAGS
flags.DEFINE_string('pretrained_model_dir', "backbone", 'which model to use.')
flags.DEFINE_string('output_dir', "output_dir", 'which model to use.')


def create_readable_names_for_imagenet_labels():
    """Create a dict mapping label id to human readable string.

    Returns:
        labels_to_names: dictionary where keys are integers from to 1000
        and values are human-readable names.

    We retrieve a synset file, which contains a list of valid synset labels used
    by ILSVRC competition. There is one synset one per line, eg.
            #   n01440764
            #   n01443537
    We also retrieve a synset_to_human_file, which contains a mapping from
    synsets to human-readable names for every synset in Imagenet.
    These are stored in a tsv format, as follows:
            #   n02119247    black fox
            #   n02119359    silver fox
    We assign each synset (in alphabetical order) an integer, starting from 1
    (since 0 is reserved for the background class).

    """

    base_url = 'http://cnbj1-fds.api.xiaomi.net/ml-datasets/imagenet/'  # noqa
    synset_url = '{}/imagenet_lsvrc_2015_synsets.txt'.format(base_url)
    synset_to_human_url = '{}/imagenet_metadata.txt'.format(base_url)

    filename, _ = urllib.urlretrieve(synset_url)
    synset_list = [s.strip() for s in open(filename).readlines()]
    num_synsets_in_ilsvrc = len(synset_list)
    assert num_synsets_in_ilsvrc == 1000

    filename, _ = urllib.urlretrieve(synset_to_human_url)
    synset_to_human_list = open(filename).readlines()
    num_synsets_in_all_imagenet = len(synset_to_human_list)
    assert num_synsets_in_all_imagenet == 21842

    synset_to_human = {}
    for s in synset_to_human_list:
        parts = s.strip().split('\t')
        assert len(parts) == 2
        synset = parts[0]
        human = parts[1]
        synset_to_human[synset] = human

    label_index = 1
    labels_to_names = {0: 'background'}
    for synset in synset_list:
        name = synset_to_human[synset]
        labels_to_names[label_index] = name
        label_index += 1

    return labels_to_names


def main(unused_argv):
    tf.reset_default_graph()

    # For simplicity we just decode jpeg inside tensorflow.
    # But one can provide any input obviously.
    file_input = tf.placeholder(tf.string, ())

    image = tf.image.decode_jpeg(tf.read_file(file_input))

    images = tf.expand_dims(image, 0)
    images = tf.cast(images, tf.float32) / 128. - 1
    images.set_shape((None, None, None, 3))
    images = tf.image.resize_images(images, (224, 224))
    mobilenet_model = MobilenetV2()

    logits, endpoints = mobilenet_model.forward(
        images,
        is_training=False)

    exclude = ['global_step']
    variables_to_restore = tf.contrib.slim.get_variables_to_restore(
        exclude=exclude)
    variables_map = {}
    for v in variables_to_restore:
        # print(v.name)
        old_name = v.name.split(':')[0]
        old_name = old_name.replace('conv2d/kernel', 'weights')
        old_name = old_name.replace('conv2d/bias', 'biases')
        variables_map[old_name] = v
    tf.train.init_from_checkpoint(FLAGS.pretrained_model_dir,
                                  variables_map)

    # download panda sample image
    #filename, _ = urllib.urlretrieve(
    #    'https://upload.wikimedia.org/wikipedia/commons/f/fe/Giant_Panda_in_Beijing_Zoo_1.JPG')  # noqa
    filename = '/home/work/liuqi/datasets/imagenet/validation/ILSVRC2012_val_00000001.JPEG'

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        x = endpoints['Predictions'].eval(
            feed_dict={file_input: filename})
        if not os.path.exists(FLAGS.output_dir):
            os.makedirs(FLAGS.output_dir)
        saver.save(sess, FLAGS.output_dir + '/mobilenet-v2-224.ckpt')

    # validation
    label_map = create_readable_names_for_imagenet_labels()
    # output: ('Top 1 prediction: ', 389, 'giant panda, panda, panda bear, coon bear, Ailuropoda melanoleuca', 0.83625776)  # noqa
    print("Top 1 prediction: ", x.argmax(), label_map[x.argmax()], x.max())


if __name__ == '__main__':
    tf.app.run()
