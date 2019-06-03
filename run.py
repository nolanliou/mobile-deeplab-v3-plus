import time
import numpy as np
import tensorflow as tf

# tf.enable_eager_execution()

from segmentation_dataset import SegmentationDataset
from deeplab_v3_plus import DeeplabV3Plus
from input_preprocess import decode_org_image
from unet import UNet
from utils import get_model_learning_rate

flags = tf.app.flags

FLAGS = flags.FLAGS

# Settings for logging.

flags.DEFINE_string('logdir', None,
                    'Where the checkpoint and logs are stored.')

flags.DEFINE_integer('log_steps', 10,
                     'Display logging information at every log_steps.')

flags.DEFINE_integer('save_interval_secs', 1200,
                     'How often, in seconds, we save the model to disk.')

# Settings for training strategy.

flags.DEFINE_enum('learning_policy', 'poly', ['poly', 'step'],
                  'Learning rate policy for training.')

# Use 0.007 when training on PASCAL augmented training set, train_aug. When
# fine-tuning on PASCAL trainval set, use learning rate=0.0001.
flags.DEFINE_float('base_learning_rate', .0001,
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
flags.DEFINE_integer('batch_size', 16,
                     'The number of images in each batch during training.')

flags.DEFINE_integer('num_clones', 1,
                     'The number of clones for multiple GPU training.')

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

flags.DEFINE_integer('decoder_output_stride', None,
                     'The ratio of input to decoder output spatial resolution.')

# Dataset settings.
flags.DEFINE_string('dataset_name', None,
                    '[pascal_voc2012|people_segmentation '
                    'Name of the segmentation dataset.')

flags.DEFINE_string('train_subset', 'trainaug',
                    'Which split of the dataset to be used for training')

flags.DEFINE_string('val_subset', 'val',
                    'Which split of the dataset to be used for training')

flags.DEFINE_string('dataset_dir', None, 'Where the dataset reside.')

# run settings.
flags.DEFINE_boolean('debug', False, 'Debug or not')

flags.DEFINE_string('backbone_type', "MobilenetV2", '[MobilenetV2|MobilenetV3]')
flags.DEFINE_string('model_type', "deeplab-v3-plus", 'which model to use.')

flags.DEFINE_string('pretrained_model_dir', "",
                    'pretrained deeplab-v3-plus model dir.')

flags.DEFINE_string('pretrained_backbone_model_dir', "",
                    'pretrained backbone(mobilnet-v2) model directory.')

flags.DEFINE_string('export_dir', "", 'Directory to export model.')

flags.DEFINE_string('mode', "train", '[train|eval|export].')

flags.DEFINE_boolean('quant_friendly', False,
                     'Remove BN and relu behind Depthwise-Convolution,'
                     ' and replace relu6 with relu.')


class TimeHistory(tf.train.SessionRunHook):
    def __init__(self):
        self.times = []
        self.iter_time_start = None

    def begin(self):
        self.times = []

    def before_run(self, run_context):
        self.iter_time_start = time.time()

    def after_run(self, run_context, run_values):
        self.times.append(time.time() - self.iter_time_start)


def segmentation_model_fn(features,
                          labels,
                          mode,
                          params):
    if isinstance(features, dict):
        features = features['features']
    num_classes = params['num_classes']
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    if params['model_type'] == 'unet':
        model = UNet(num_classes=num_classes)
        logits = model.forward_pass(features,
                                    is_training=is_training)
    else:
        pretrained_model_dir = params.get('pretrained_model_dir', '')
        pretrained_backbone_model_dir = ''
        if not pretrained_model_dir and is_training:
            pretrained_backbone_model_dir = params.get(
                'pretrained_backbone_model_dir', '')
        model = DeeplabV3Plus(
            num_classes=num_classes,
            backbone=params['backbone_type'],
            pretrained_backbone_model_dir=pretrained_backbone_model_dir,
            model_input_size=params['model_input_size'],
            atrous_rates=params['atrous_rates'],
            output_stride=params['output_stride'],
            weight_decay=params['weight_decay'],
            decoder_output_stride=params['decoder_output_stride'],
            quant_friendly=params['quant_friendly'])

        logits = model.forward(features,
                               is_training=is_training)
        if is_training and pretrained_model_dir:
            print('Init from pretrained deeplab model')
            exclude = ['global_step']
            variables_to_restore = tf.contrib.slim.get_variables_to_restore(
                exclude=exclude)
            variables_map = {}
            for v in variables_to_restore:
                variables_map[v.name.split(':')[0]] = v
            tf.train.init_from_checkpoint(pretrained_model_dir,
                                          variables_map)
    pred_labels = tf.expand_dims(
        tf.argmax(logits, axis=3, output_type=tf.int32), axis=3, name='Output')

    # Predict
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'pred_classes': pred_labels,
        }
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions
        )

    one_hot_labels = tf.one_hot(labels, depth=num_classes,
                                on_value=1.0, off_value=0.0, axis=-1)

    logits_by_num_classes = tf.reshape(logits, [-1, num_classes])
    labels_by_num_classes = tf.reshape(one_hot_labels, [-1, num_classes])
    ignore_label = params['ignore_label']

    labels_flat = tf.reshape(labels, [-1])
    not_ignore_mask = tf.to_float(
        tf.not_equal(labels_flat, ignore_label)) * 1.0

    valid_indices = tf.to_int32(labels_flat <= (num_classes - 1))

    valid_labels = tf.dynamic_partition(labels_flat, valid_indices,
                                        num_partitions=2)[1]
    valid_preds = tf.dynamic_partition(tf.reshape(pred_labels, [-1]),
                                       valid_indices,
                                       num_partitions=2)[1]
    labels_flat = tf.reshape(valid_labels, [-1])
    pred_labels_flat = tf.reshape(valid_preds, [-1])

    with tf.name_scope('loss'):
        cross_entropy = tf.losses.softmax_cross_entropy(
            onehot_labels=labels_by_num_classes,
            logits=logits_by_num_classes,
            weights=not_ignore_mask)

        # Create a tensor named cross_entropy for logging purposes.
        tf.identity(cross_entropy, name='cross_entropy')
        tf.summary.scalar('cross_entropy', cross_entropy)

        total_loss = cross_entropy + tf.add_n(model.losses())
        tf.identity(total_loss, name='total_loss')
        tf.summary.scalar('total_loss', total_loss)

    with tf.name_scope('accuracy'):
        accuracy = tf.metrics.accuracy(
            labels_flat, pred_labels_flat)
        mean_iou = tf.metrics.mean_iou(labels_flat, pred_labels_flat,
                                       num_classes)
    metrics = {'pixel_accuracy': accuracy, 'mean_iou': mean_iou}

    # evaluation
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=cross_entropy,
            eval_metric_ops=metrics,
            evaluation_hooks=None,
        )

    images = tf.cast(
        tf.map_fn(decode_org_image, features),
        tf.uint8)
    # Scale up summary image pixel values for better visualization.
    pixel_scaling = max(1, 255 // params['num_classes'])
    summary_label = tf.cast(labels * pixel_scaling, tf.uint8)

    summary_pred_labels = tf.cast(pred_labels * pixel_scaling, tf.uint8)
    tf.summary.image('samples/image', images)
    tf.summary.image('samples/label', summary_label)
    tf.summary.image('samples/prediction', summary_pred_labels)

    global_step = tf.train.get_or_create_global_step()
    learning_rate = get_model_learning_rate(
        global_step,
        params['learning_policy'],
        params['base_learning_rate'],
        params['learning_rate_decay_step'],
        params['learning_rate_decay_factor'],
        params['training_number_of_steps'],
        params['learning_power'],
        params['slow_start_step'],
        params['slow_start_learning_rate'])
    tf.identity(learning_rate, name='learning_rate')
    tf.summary.scalar('learning_rate', learning_rate)

    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                           momentum=params['momentum'])
    # Batch norm requires update ops to be added as a dependency to
    # the train_op
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(total_loss, global_step)

    def compute_mean_iou(total_cm, name='mean_iou'):
        """Compute the mean intersection-over-union via the confusion matrix."""
        sum_over_row = tf.to_float(tf.reduce_sum(total_cm, 0))
        sum_over_col = tf.to_float(tf.reduce_sum(total_cm, 1))
        cm_diag = tf.to_float(tf.diag_part(total_cm))
        denominator = sum_over_row + sum_over_col - cm_diag

        # The mean is only computed over classes that appear in the
        # label or prediction tensor. If the denominator is 0, we need to
        # ignore the class.
        num_valid_entries = tf.reduce_sum(tf.cast(
            tf.not_equal(denominator, 0), dtype=tf.float32))

        # If the value of the denominator is 0, set it to 1 to avoid
        # zero division.
        denominator = tf.where(
            tf.greater(denominator, 0),
            denominator,
            tf.ones_like(denominator))
        iou = tf.div(cm_diag, denominator)

        for i in range(params['num_classes']):
            tf.identity(iou[i], name='train_iou_class{}'.format(i))
            tf.summary.scalar('train_iou_class{}'.format(i), iou[i])

        # If the number of valid entries is 0 (no classes) we return 0.
        result = tf.where(
            tf.greater(num_valid_entries, 0),
            tf.reduce_sum(iou, name=name) / num_valid_entries,
            0)
        return result

    train_mean_iou = compute_mean_iou(mean_iou[1])

    tf.identity(train_mean_iou, name='train_mean_iou')
    tf.summary.scalar('train_mean_iou', train_mean_iou)

    tensors_to_log = {
        'learning_rate': learning_rate,
        'cross_entropy': cross_entropy,
        'total_loss': total_loss,
        'train_pixel_accuracy': accuracy[1],
        'train_mean_iou': train_mean_iou,
    }

    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=10)
    train_hooks = [logging_hook]

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=total_loss,
        train_op=train_op,
        training_hooks=train_hooks
    )


def train():
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
    eval_dataset = SegmentationDataset(
        FLAGS.dataset_name,
        FLAGS.dataset_dir,
        FLAGS.val_subset,
        FLAGS.model_input_size[0],
        FLAGS.model_input_size[1],
        is_training=False)

    num_train_data = train_dataset.get_num_data()
    training_num_of_steps = FLAGS.training_number_of_steps
    if FLAGS.num_clones > 1:
        training_num_of_steps = \
            (training_num_of_steps + FLAGS.num_clones - 1) / FLAGS.num_clones
    epochs = (training_num_of_steps * FLAGS.batch_size * FLAGS.num_clones
              + num_train_data - 1) // num_train_data
    save_checkpoints_steps = \
        (num_train_data + FLAGS.batch_size - 1) // FLAGS.batch_size
    if FLAGS.num_clones > 1:
        strategy = tf.contrib.distribute.MirroredStrategy(
            num_gpus=FLAGS.num_clones)
        run_config = tf.estimator.RunConfig(
            save_checkpoints_steps=save_checkpoints_steps,
            train_distribute=strategy,
        )
    else:
      run_config = tf.estimator.RunConfig(
          save_checkpoints_steps=save_checkpoints_steps)
    pretrained_model_dir = FLAGS.pretrained_model_dir
    pretrained_backbone_model_dir = FLAGS.pretrained_backbone_model_dir
    if tf.train.latest_checkpoint(FLAGS.logdir):
        tf.logging.info('Ignore initialization; other checkpoint exists')
        pretrained_model_dir = ''
        pretrained_backbone_model_dir = ''
    model = tf.estimator.Estimator(
        model_fn=segmentation_model_fn,
        model_dir=FLAGS.logdir,
        config=run_config,
        params={
            'model_type': FLAGS.model_type,
            'backbone_type': FLAGS.backbone_type,
            'logdir': FLAGS.logdir,
            'pretrained_model_dir': pretrained_model_dir,
            'pretrained_backbone_model_dir': pretrained_backbone_model_dir,
            'num_classes': train_dataset.get_num_classes(),
            'ignore_label': train_dataset.get_ignore_label(),
            'model_input_size': FLAGS.model_input_size,
            'output_stride': FLAGS.output_stride,
            'decoder_output_stride': FLAGS.decoder_output_stride,
            'atrous_rates': FLAGS.atrous_rates,
            'weight_decay': FLAGS.weight_decay,
            'batch_size': FLAGS.batch_size,
            'learning_policy': FLAGS.learning_policy,
            'base_learning_rate': FLAGS.base_learning_rate,
            'learning_rate_decay_step': FLAGS.learning_rate_decay_step,
            'learning_rate_decay_factor': FLAGS.learning_rate_decay_factor,
            'training_number_of_steps': training_num_of_steps,
            'learning_power': FLAGS.learning_power,
            'slow_start_step': FLAGS.slow_start_step,
            'slow_start_learning_rate': FLAGS.slow_start_learning_rate,
            'momentum': FLAGS.momentum,
            'quant_friendly': FLAGS.quant_friendly,
        }
    )
    time_hist = TimeHistory()
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: train_dataset.make_batch(FLAGS.batch_size,
                                                  epochs,
                                                  FLAGS.num_clones),
        max_steps=training_num_of_steps,
        hooks=[time_hist])
    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: eval_dataset.make_batch(1),
        steps=eval_dataset.get_num_data())
    tf.estimator.train_and_evaluate(model, train_spec, eval_spec)

    total_time = sum(time_hist.times)
    print("total time with %d GPUs: %f seconds" % (FLAGS.num_clones,
                                                   total_time))

    avg_time_per_batch = np.mean(time_hist.times)
    print("%f images/second with %d GPUs" %
          (FLAGS.batch_size * FLAGS.num_clones / avg_time_per_batch,
           FLAGS.num_clones))


def evaluate():
    eval_dataset = SegmentationDataset(
        FLAGS.dataset_name,
        FLAGS.dataset_dir,
        FLAGS.val_subset,
        FLAGS.model_input_size[0],
        FLAGS.model_input_size[1],
        is_training=False)

    run_config = tf.estimator.RunConfig()
    model = tf.estimator.Estimator(
        model_fn=segmentation_model_fn,
        model_dir=FLAGS.logdir,
        config=run_config,
        params={
            'model_type': FLAGS.model_type,
            'backbone_type': FLAGS.backbone_type,
            'logdir': FLAGS.logdir,
            'num_classes': eval_dataset.get_num_classes(),
            'ignore_label': eval_dataset.get_ignore_label(),
            'model_input_size': FLAGS.model_input_size,
            'atrous_rates': FLAGS.atrous_rates,
            'output_stride': FLAGS.output_stride,
            'decoder_output_stride': FLAGS.decoder_output_stride,
            'weight_decay': FLAGS.weight_decay,
            'quant_friendly': FLAGS.quant_friendly,
        }
    )

    # Evaluate the model and print results
    eval_results = model.evaluate(
        # Batch size must be 1 for testing because the images' size differs
        input_fn=lambda: eval_dataset.make_batch(1),
        steps=eval_dataset.get_num_data())
    print(eval_results)


def export_model():
    eval_dataset = SegmentationDataset(
        FLAGS.dataset_name,
        FLAGS.dataset_dir,
        FLAGS.val_subset,
        FLAGS.model_input_size[0],
        FLAGS.model_input_size[1],
        is_training=False)

    run_config = tf.estimator.RunConfig()
    estimator = tf.estimator.Estimator(
        model_fn=segmentation_model_fn,
        model_dir=FLAGS.logdir,
        config=run_config,
        params={
            'model_type': FLAGS.model_type,
            'backbone_type': FLAGS.backbone_type,
            'logdir': FLAGS.logdir,
            'num_classes': eval_dataset.get_num_classes(),
            'ignore_label': eval_dataset.get_ignore_label(),
            'model_input_size': FLAGS.model_input_size,
            'atrous_rates': FLAGS.atrous_rates,
            'output_stride': FLAGS.output_stride,
            'decoder_output_stride': FLAGS.decoder_output_stride,
            'weight_decay': FLAGS.weight_decay,
            'quant_friendly': FLAGS.quant_friendly,
        }
    )

    # Export the model
    def make_serving_input_receiver_fn():
        inputs = {'features':
            tf.placeholder(shape=[None, FLAGS.model_input_size[0],
                                  FLAGS.model_input_size[1], 3],
                           dtype=tf.float32,
                           name='Input')}
        return tf.estimator.export.build_raw_serving_input_receiver_fn(inputs)

    print('Export model to %s' % FLAGS.export_dir)
    tf.gfile.DeleteRecursively(FLAGS.export_dir)
    tf.gfile.MakeDirs(FLAGS.export_dir)
    estimator.export_savedmodel(
        export_dir_base=FLAGS.export_dir,
        serving_input_receiver_fn=make_serving_input_receiver_fn())


def main(unused_argv):
    if FLAGS.model_type not in ['unet', 'deeplab-v3-plus']:
        raise ValueError('Only support unet and deeplab-v3-plus but got ',
                         FLAGS.model_type)
    if FLAGS.backbone_type not in ['MobilenetV2', 'MobilenetV3']:
        raise ValueError('Only support MobilenetV2 and MobilenetV3 but got ',
                         FLAGS.backbone_type)

    tf.logging.set_verbosity(tf.logging.INFO)

    if FLAGS.mode == 'train':
        train()
    elif FLAGS.mode == 'eval':
        evaluate()
    elif FLAGS.mode == 'export':
        export_model()
    else:
        raise ValueError('Only support train and eval but got ',
                         FLAGS.model_type)


if __name__ == '__main__':
    flags.mark_flag_as_required('dataset_dir')
    flags.mark_flag_as_required('dataset_name')
    tf.app.run()
