import tensorflow as tf
import onnxruntime as ort

class optimizer:
    @staticmethod
    def convert_to_onnx(model, output_path):
        tf.saved_model.load(model, "./saved_model")
        !python -m tf2onnx.convert \
            --saved-model /tmp/tf_model \
            --output {output_path} \
            --opset 13

    @staticmethod
    def build_ort_session(onnx_model_path):
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return ort.InferenceSession(onnx_model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"], provider_options=[{"device_id": 0, "cudnn_conv_algo_search": "DEFAULT"}, {}])
       