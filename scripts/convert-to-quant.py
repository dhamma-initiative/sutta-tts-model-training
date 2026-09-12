# works with pip install onnxruntime==1.18.1 onnx==1.16.1
from onnxruntime.quantization import QuantType, quantize_dynamic
quantize_dynamic(
    model_input='./en_gb-suttaplayer-medium.onnx',
    model_output='./en_gb-suttaplayer-medium-quant.onnx',
    weight_type=QuantType.QUInt8
)