import onnxruntime as ort
import tensorflow as tf

def convert_to_onnx(model, output_path):
    tf.saved_model.load(model, "./saved_model")
    !python -m tf2onnx.convert \
        --saved-model /tmp/tf_model \
        --output {output_path} \
        --opset 13

    
class inference_profiler:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"], provider_options=[{"device_id": 0, "cudnn_conv_algo_search": "DEFAULT"}, {}])

    def run(self, input_data):
        name = self.session.get_inputs()[0].name
        return self.session.run(None, {name: input_data})