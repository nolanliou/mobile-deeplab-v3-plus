import os, argparse

import tensorflow as tf

# The original freeze_graph function
from tensorflow.python.tools.freeze_graph import freeze_graph 
from tensorflow.python.saved_model import tag_constants


def freeze_graph_func(model_dir, output_node_names, output_dir):
    """Extract the sub graph defined by the output nodes and convert 
    all its variables into constant 

    Args:
        model_dir: the root folder containing the checkpoint state file
        output_node_names: a string, containing all the output node's names, 
                            comma separated
    """
    if not tf.gfile.Exists(model_dir):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export "
            "directory: %s" % model_dir)

    if not output_node_names:
        print("You need to supply the name of a node to --output_node_names.")
        return -1

    sub_dirs = [name for name in os.listdir(model_dir)
         if os.path.isdir(os.path.join(model_dir, name))]
    model_dir = os.path.join(model_dir, sub_dirs[0])

    output_graph_filename = os.path.join(output_dir, 'frozen_model.pb')
    initializer_nodes = ''
    freeze_graph(
        input_graph=None,
        input_saver=False,
        input_binary=False,
        input_checkpoint=None,
        output_node_names=output_node_names,
        restore_op_name=None,
        filename_tensor_name=None,
        output_graph=output_graph_filename,
        clear_devices=True,
        initializer_nodes=initializer_nodes,
        input_meta_graph=False,
        input_saved_model_dir=model_dir,
        saved_model_tags=tag_constants.SERVING)
    print('model has been frozen!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="", help="Model folder to export")
    parser.add_argument("--output_node_names", type=str, default="", help="The name of the output nodes, comma separated.")
    parser.add_argument("--output_dir", type=str, default="", help="output directory to keep frozen model.")
    args = parser.parse_args()

    freeze_graph_func(args.model_dir, args.output_node_names, args.output_dir)
