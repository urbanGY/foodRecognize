
# python optimize_for_inference.py --input=./model/output_graph.pb --output=./model/opt_graph.pb --input_names="DecodeJpeg/contents" --output_names="final_result"

# IMAGE_SIZE=299
# toco --input_file=./model/opt_graph.pb --output_file=./model/lite_graph.tflite --input_format=TENSORFLOW_GRAPHDEF --output_format=TFLITE --input_shape=1,299,299,3 --input_array=DecodeJpeg/contents --output_array=final_result --inference_type=FLOAT --input_type=FLOAT --enable_select_tf_ops=true
# bazel run --config=opt //tensorflow/contrib/lite/toco:toco -- --input_file=./model/opt_graph.pb --output_file=./model/lite_graph.tflite --input_format=TENSORFLOW_GRAPHDEF --output_format=TFLITE --input_shape=1,299,299,3 --input_array=DecodeJpeg/contents --output_array=final_result --inference_type=FLOAT --input_type=FLOAT
# tflite_convert --output_file=./model/lite_graph.tflite --graph_def_file=./model/output_graph.pb --enable_v1_converter --input_arrays=DecodeJpeg/contents --output_arrays=final_result --allow_custom_ops

# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
# saved_model_cli show --dir ./saved_model.pb --all
# saved_model_dir = '.'
# converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
#                                        tf.lite.OpsSet.SELECT_TF_OPS]
# tflite_model = converter.convert()
# open('./converted_model.tflite', 'wb').write(tflite_model)

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

graph_def_file = './model/output_graph.pb'
input_arrays = ["Sub"]
output_arrays = ["final_result"]
tflite_file = './model/lite_graph_2.tflite'

converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays)
tflite_model = converter.convert()
open(tflite_file, "wb").write(tflite_model)
