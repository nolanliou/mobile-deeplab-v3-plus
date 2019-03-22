import tensorflow as tf
from tensorflow.python import debug as tf_debug

import deeplab_v3_plus

# tf.enable_eager_execution()

from segmentation_dataset import SegmentationDataset

flags = tf.app.flags

FLAGS = flags.FLAGS

# Settings for logging.

flags.DEFINE_string('train_logdir', None,
                    'Where the checkpoint and logs are stored.')

flags.DEFINE_integer('log_steps', 10,
                     'Display logging information at every log_steps.')

flags.DEFINE_integer('save_interval_secs', 1200,
                     'How often, in seconds, we save the model to disk.')

flags.DEFINE_integer('save_summaries_secs', 600,
                     'How often, in seconds, we compute the summaries.')

flags.DEFINE_boolean('save_summaries_images', False,
                     'Save sample inputs, labels, and semantic predictions as '
                     'images to summary.')

# Settings for training strategy.

flags.DEFINE_enum('learning_policy', 'poly', ['poly', 'step'],
                  'Learning rate policy for training.')

# Use 0.007 when training on PASCAL augmented training set, train_aug. When
# fine-tuning on PASCAL trainval set, use learning rate=0.0001.
flags.DEFINE_float('base_learning_rate', .0007,
                   'The base learning rate for model training.')

flags.DEFINE_float('learning_rate_decay_factor', 0.1,
                   'The rate to decay the base learning rate.')

flags.DEFINE_integer('learning_rate_decay_step', 2000,
                     'Decay the base learning rate at a fixed step.')

flags.DEFINE_float('learning_power', 0.9,
                   'The power value used in the poly learning policy.')

flags.DEFINE_integer('training_number_of_steps', 30000,
                     'The number of steps used for training')

flags.DEFINE_float('momentum', 0.9, 'The momentum value to use')

# When fine_tune_batch_norm=True, use at least batch size larger than 12
# (batch size more than 16 is better). Otherwise, one could use smaller batch
# size and set fine_tune_batch_norm=False.
flags.DEFINE_integer('batch_size', 8,
                     'The number of images in each batch during training.')

flags.DEFINE_integer('train_epochs', 23,
                     help='Number of training epochs: '
                          'For 30K iteration with batch size 6, train_epoch = 17.01 (= 30K * 6 / 10,582). '
                          'For 30K iteration with batch size 8, train_epoch = 22.68 (= 30K * 8 / 10,582). '
                          'For 30K iteration with batch size 10, train_epoch = 25.52 (= 30K * 10 / 10,582). '
                          'For 30K iteration with batch size 11, train_epoch = 31.19 (= 30K * 11 / 10,582). '
                          'For 30K iteration with batch size 15, train_epoch = 42.53 (= 30K * 15 / 10,582). '
                          'For 30K iteration with batch size 16, train_epoch = 45.36 (= 30K * 16 / 10,582).')

flags.DEFINE_integer('epochs_per_eval', 1,
                     'The number of epochs to run between evaluation.')

# For weight_decay, use 0.00004 for MobileNet-V2 or Xcpetion model variants.
# Use 0.0001 for ResNet model variants.
flags.DEFINE_float('weight_decay', 0.00004,
                   'The value of the weight decay for training.')

flags.DEFINE_multi_integer('model_input_size', [513, 513],
                           'Image crop size [height, width] during training.')

flags.DEFINE_float('last_layer_gradient_multiplier', 1.0,
                   'The gradient multiplier for last layers, which is used to '
                   'boost the gradient of last layers if the value > 1.')

flags.DEFINE_boolean('upsample_logits', True,
                     'Upsample logits during training.')

# Set to False if one does not want to re-use the trained classifier weights.
flags.DEFINE_boolean('initialize_last_layer', True,
                     'Initialize the last layer.')

flags.DEFINE_boolean('last_layers_contain_logits_only', False,
                     'Only consider logits as last layers or not.')

flags.DEFINE_integer('slow_start_step', 0,
                     'Training model with small learning rate for few steps.')

flags.DEFINE_float('slow_start_learning_rate', 1e-4,
                   'Learning rate employed during slow start.')

flags.DEFINE_float('min_scale_factor', 0.5,
                   'Mininum scale factor for data augmentation.')

flags.DEFINE_float('max_scale_factor', 2.,
                   'Maximum scale factor for data augmentation.')

flags.DEFINE_float('scale_factor_step_size', 0.25,
                   'Scale factor step size for data augmentation.')

# For `xception_65`, use atrous_rates = [12, 24, 36] if output_stride = 8, or
# rates = [6, 12, 18] if output_stride = 16. For `mobilenet_v2`, use None. Note
# one could use different atrous_rates/output_stride during training/evaluation.
flags.DEFINE_multi_integer('atrous_rates', None,
                           'Atrous rates for atrous spatial pyramid pooling.')

flags.DEFINE_integer('output_stride', 16,
                     'The ratio of input to output spatial resolution.')

# Dataset settings.
flags.DEFINE_string('dataset_name', 'pascal_voc2012',
                    'Name of the segmentation dataset.')

flags.DEFINE_string('train_subset', 'train',
                    'Which split of the dataset to be used for training')

flags.DEFINE_string('val_subset', 'val',
                    'Which split of the dataset to be used for training')

flags.DEFINE_string('dataset_dir', None, 'Where the dataset reside.')

flags.DEFINE_boolean('debug', False, 'Debug or not')


def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)

    train_dataset = SegmentationDataset(
        FLAGS.dataset_name,
        FLAGS.dataset_dir,
        FLAGS.train_subset,
        FLAGS.model_input_size[0],
        FLAGS.model_input_size[1],
        FLAGS.min_scale_factor,
        FLAGS.max_scale_factor,
        FLAGS.scale_factor_step_size,
        is_training=True)
    val_dataset = SegmentationDataset(
        FLAGS.dataset_name,
        FLAGS.dataset_dir,
        FLAGS.val_subset,
        FLAGS.model_input_size[0],
        FLAGS.model_input_size[1],
        is_training=False)
    run_config = tf.estimator.RunConfig()
    model = tf.estimator.Estimator(
        model_fn=deeplab_v3_plus.deeplab_v3_plus_model_fn,
        model_dir=FLAGS.train_logdir,
        config=run_config,
        params={
            'num_classes': train_dataset.get_num_classes(),
            'model_input_size': FLAGS.model_input_size,
            'output_stride': FLAGS.output_stride,
            'weight_decay': FLAGS.weight_decay,
            'batch_size': FLAGS.batch_size,
            'learning_policy': FLAGS.learning_policy,
            'base_learning_rate': FLAGS.base_learning_rate,
            'learning_rate_decay_step': FLAGS.learning_rate_decay_step,
            'learning_rate_decay_factor': FLAGS.learning_rate_decay_factor,
            'training_number_of_steps': FLAGS.training_number_of_steps,
            'learning_power': FLAGS.learning_power,
            'slow_start_step': FLAGS.slow_start_step,
            'slow_start_learning_rate': FLAGS.slow_start_learning_rate,
            'momentum': FLAGS.momentum,
        }
    )

    for _ in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
        eval_hooks = None

        if FLAGS.debug:
            debug_hook = tf_debug.LocalCLIDebugHook()
            eval_hooks = [debug_hook]

        print("Start training.")
        model.train(
            input_fn=lambda: train_dataset.make_batch(FLAGS.batch_size,
                                                      FLAGS.epochs_per_eval),
            # steps=1 # For debug
        )

        print("Start evaluation.")
        # Evaluate the model and print results
        eval_results = model.evaluate(
            # Batch size must be 1 for testing because the images' size differs
            input_fn=lambda: val_dataset.make_batch(FLAGS.batch_size),
            hooks=eval_hooks,
            # steps=1  # For debug
        )
        print(eval_results)


if __name__ == '__main__':
    flags.mark_flag_as_required('train_logdir')
    flags.mark_flag_as_required('dataset_dir')
    tf.app.run()
