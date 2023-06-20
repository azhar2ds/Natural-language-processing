
import os

# pylint: disable=g-import-not-at-top
if not os.path.splitext(__file__)[0].endswith(
    os.path.join("tflite_runtime", "analyzer")):
  # This file is part of tensorflow package.
  from tensorflow.lite.python import wrap_toco
  from tensorflow.lite.python.analyzer_wrapper import _pywrap_analyzer_wrapper as _analyzer_wrapper
  from tensorflow.python.util.tf_export import tf_export as _tf_export
else:
  # This file is part of tflite_runtime package.
  from tflite_runtime import _pywrap_analyzer_wrapper as _analyzer_wrapper

  def _tf_export(*x, **kwargs):
    del x, kwargs
    return lambda x: x


@_tf_export("lite.experimental.Analyzer")
class ModelAnalyzer():

  @staticmethod
  def analyze(model_path=None,
              model_content=None,
              gpu_compatibility=False,
              **kwargs):

    if not model_path and not model_content:
      raise ValueError("neither `model_path` nor `model_content` is provided")
    if model_path:
      print(f"=== {model_path} ===\n")
      tflite_model = model_path
      input_is_filepath = True
    else:
      print("=== TFLite ModelAnalyzer ===\n")
      tflite_model = model_content
      input_is_filepath = False

    if kwargs.get("experimental_use_mlir", False):
      print(
          wrap_toco.wrapped_flat_buffer_file_to_mlir(tflite_model,
                                                     input_is_filepath))
    else:
      print(
          _analyzer_wrapper.ModelAnalyzer(tflite_model, input_is_filepath,
                                          gpu_compatibility))
