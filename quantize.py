# /*---------------------------------------------------------------------------------------------
#  * Copyright (c) 2024 STMicroelectronics.
#  * All rights reserved.
#  *
#  * This software is licensed under terms that can be found in the LICENSE file in
#  * the root directory of this software component.
#  * If no LICENSE file comes with this software, it is provided AS-IS.
#  *--------------------------------------------------------------------------------------------*/

from pathlib import Path
import onnx
import numpy as np
from onnxruntime.quantization import quantize_static, CalibrationMethod, QuantFormat, QuantType
from onnxruntime.quantization import CalibrationDataReader
from onnxruntime.quantization.shape_inference import quant_pre_process
from onnxruntime import InferenceSession
import math

model = onnx.load("./q_weights/model_ctc.onnx")
encoder_dim = 64

ops = sorted(set(node.op_type for node in model.graph.node))

for node in model.graph.node:
    if node.op_type == "MatMul":
        print(node.name)
node_to_exclude = [
# "/spec_encoder/input_projection/input_projection.0/linear/MatMul",
# "/spec_encoder/layers.0/sequential/sequential.0/module/sequential/sequential.0/MatMul",
# "/spec_encoder/layers.0/sequential/sequential.0/module/sequential/sequential.3/MatMul",
# "/spec_encoder/layers.0/sequential/sequential.1/module/attention/query_proj/MatMul",
# "/spec_encoder/layers.0/sequential/sequential.1/module/attention/key_proj/MatMul",
# "/spec_encoder/layers.0/sequential/sequential.1/module/attention/value_proj/MatMul",
# "/spec_encoder/layers.0/sequential/sequential.1/module/attention/pos_proj/MatMul",
# "/spec_encoder/layers.0/sequential/sequential.1/module/attention/MatMul",
# "/spec_encoder/layers.0/sequential/sequential.1/module/attention/MatMul_1",
# "/spec_encoder/layers.0/sequential/sequential.1/module/attention/MatMul_2",
# "/spec_encoder/layers.0/sequential/sequential.1/module/attention/out_proj/MatMul",
# "/spec_encoder/layers.0/sequential/sequential.3/module/sequential/sequential.0/MatMul",
# "/spec_encoder/layers.0/sequential/sequential.3/module/sequential/sequential.3/MatMul",
# "/spec_encoder/layers.1/sequential/sequential.0/module/sequential/sequential.0/MatMul",
# "/spec_encoder/layers.1/sequential/sequential.0/module/sequential/sequential.3/MatMul",
# "/spec_encoder/layers.1/sequential/sequential.1/module/attention/query_proj/MatMul",
# "/spec_encoder/layers.1/sequential/sequential.1/module/attention/key_proj/MatMul",
# "/spec_encoder/layers.1/sequential/sequential.1/module/attention/value_proj/MatMul",
# "/spec_encoder/layers.1/sequential/sequential.1/module/attention/pos_proj/MatMul",
# "/spec_encoder/layers.1/sequential/sequential.1/module/attention/MatMul",
# "/spec_encoder/layers.1/sequential/sequential.1/module/attention/MatMul_1",
# "/spec_encoder/layers.1/sequential/sequential.1/module/attention/MatMul_2",
# "/spec_encoder/layers.1/sequential/sequential.1/module/attention/out_proj/MatMul",
# "/spec_encoder/layers.1/sequential/sequential.3/module/sequential/sequential.0/MatMul",
# "/spec_encoder/layers.1/sequential/sequential.3/module/sequential/sequential.3/MatMul",
# "/spec_encoder/layers.2/sequential/sequential.0/module/sequential/sequential.0/MatMul",
# "/spec_encoder/layers.2/sequential/sequential.0/module/sequential/sequential.3/MatMul",
# "/spec_encoder/layers.2/sequential/sequential.1/module/attention/query_proj/MatMul",
# "/spec_encoder/layers.2/sequential/sequential.1/module/attention/key_proj/MatMul",
# "/spec_encoder/layers.2/sequential/sequential.1/module/attention/value_proj/MatMul",
# "/spec_encoder/layers.2/sequential/sequential.1/module/attention/pos_proj/MatMul",
# "/spec_encoder/layers.2/sequential/sequential.1/module/attention/MatMul",
# "/spec_encoder/layers.2/sequential/sequential.1/module/attention/MatMul_1",
# "/spec_encoder/layers.2/sequential/sequential.1/module/attention/MatMul_2",
# "/spec_encoder/layers.2/sequential/sequential.1/module/attention/out_proj/MatMul",
# "/spec_encoder/layers.2/sequential/sequential.3/module/sequential/sequential.0/MatMul",
# "/spec_encoder/layers.2/sequential/sequential.3/module/sequential/sequential.3/MatMul",
# "/head/MatMul",
]

op_types_to_quantize = []
print("Operators used:")
for op in ops:
    # if not op in ["MatMul"]:
    op_types_to_quantize.append(op)

original_onnx = "./q_weights/model_ctc.onnx"

def _preprocess_images():
    unconcatenated_batch_data = []
    root = Path(r"ken")
    paths = root.rglob(f"*npz")
    for p in paths:
        id = p.stem[2:5]
        seg = int(p.stem[9:12])
        if not (id in ["1_1", "1_2", "1_3", "1_6"]) or seg >= 100:
            continue

        data = np.load(p)
        log_mel = data["mel"][None, :35, :].astype(np.float32)

        unconcatenated_batch_data.append(log_mel)
    batch_data = np.concatenate(unconcatenated_batch_data, axis=0)

    batch_data_pos = np.zeros((batch_data.shape[0], 21, encoder_dim), dtype=np.float32)
    pe = np.zeros((10000, encoder_dim), dtype=np.float32)
    position = np.arange(0, 10000, dtype=np.float32)[:, None]
    div_term = np.exp(np.arange(0, encoder_dim, 2, dtype=np.float32) * -(math.log(10000.0) / encoder_dim))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    batch_data_pos[:] = pe[:batch_data_pos.shape[1], :]

    return {'inputs': batch_data[:10]}
    # return {'inputs': batch_data, 'inputs_pos': batch_data_pos}

class XXXDataReader(CalibrationDataReader):
    def __init__(self, model_path: str, input_name: str):
        self.enum_data = None

        # Use inference session to get input shape.
        # session = InferenceSession(model_path, None)
        # (_, height, width) = session.get_inputs()[0].shape

        # Convert image to input data
        self.nhwc_data_list = _preprocess_images()
      
        self.input_name = input_name
        self.datasize = self.nhwc_data_list['inputs'].shape[0]

        self.cur_idx = 0

    def get_next(self):
        if self.cur_idx >= self.datasize:
            return None

        self.enum_data = {
            name: nhwc_data[self.cur_idx:self.cur_idx+1]
            for name, nhwc_data in self.nhwc_data_list.items()
        }

        self.cur_idx += 1
        return self.enum_data

    def rewind(self):
        self.enum_data = None

class SEONNXPTQQuantizer:
    '''Post-training quantizer for ONNX speech enhancement models.

    Notes
    -----
    Wraps ONNX Runtime static quantization (QDQ, INT8) for single-input models.
    Uses a PyTorch `DataLoader` via `DataLoaderDataReader` for calibration data.
    '''
    def __init__(self):
        '''Initialize the ONNX PTQ quantizer.

        Parameters
        ----------
        cfg, object : User configuration with quantization parameters and output directory settings.
        model, onnxruntime.InferenceSession : Float ONNX model session to be quantized.
        dataloaders, dict[str, torch.utils.data.DataLoader] : Contains `quant_dl` for calibration samples (batch size 1).

        Notes
        -----
        - Expects single-input ONNX models; multi-input is not supported.
        '''

        # self.model is passed by the get_model API and should be an onnxruntime.InferenceSession
        # We need the actual model to infer the name of input/output nodes and perform some checks
        self.float_onnx_model = onnx.load(original_onnx)

        # Get input/output node names
        # Output node name is unused for now
        output_nodes =[node.name for node in self.float_onnx_model.graph.output]
        input_all = [node.name for node in self.float_onnx_model.graph.input]
        input_initializer =  [node.name for node in self.float_onnx_model.graph.initializer]
        net_input_nodes = list(set(input_all)  - set(input_initializer))

        self.data_reader = XXXDataReader(original_onnx, net_input_nodes[0])
        
    def quantize(self):
        '''Run ONNX quantization preprocessing, quantization, and optional static-shape conversion.

        Notes
        -----
        - Preprocesses the float ONNX model with `quant_pre_process` for shape inference.
        - Quantizes with `quantize_static` using QDQ format and INT8 weights/activations.
        - Cleans up opset imports to match the original model.
        - Fixes a dynamic sequence axis to a static value and repairs output shapes.

        Returns
        -------
        tuple : (`quantized_model_session`, `quantized_static_model_session`)
            - `quantized_model_session`, onnxruntime.InferenceSession : Quantized ONNX with dynamic input shape.
            - `quantized_static_model_session`, onnxruntime.InferenceSession : Quantized ONNX with fixed sequence length.
        '''

        # Preprocess the float model
        
        onnx_prep_path = "q_weights/preprocessed_model.onnx"
        model = onnx.load(original_onnx)
        quant_pre_process(input_model=original_onnx, output_model_path=onnx_prep_path)
        print(f"[INFO] Saved preprocessed float ONNX model at {onnx_prep_path}")

        onnx_prep_model = onnx.load(onnx_prep_path)

        # Get original model's opsets
        orig_opsets = self.float_onnx_model.opset_import
        # Remove superfluous ONNX opsets added by the preprocessing step
        del onnx_prep_model.opset_import[:]
        for op in orig_opsets:
            opset = onnx_prep_model.opset_import.add()
            opset.domain = op.domain
            opset.version = op.version

        print("Opset imports after cleanup :")
        for opset in onnx_prep_model.opset_import:
            print("opset domain=%r version=%r" % (opset.domain, opset.version))

        onnx.save(onnx_prep_model, onnx_prep_path)

        quantized_model_path = "q_weights/quantized_model_ctc.onnx"
        per_channel = False
        
        quantize_static(
            onnx_prep_path,
            quantized_model_path,
            self.data_reader,
            op_types_to_quantize=op_types_to_quantize,
            nodes_to_exclude=node_to_exclude,
            calibrate_method=CalibrationMethod.MinMax, 
            quant_format=QuantFormat.QDQ,
            per_channel=per_channel,
            weight_type=QuantType.QInt8,
            activation_type = QuantType.QInt8,
            reduce_range=False,
        )
        
        print("[INFO] Successfully converted quantized model to static input shape")

        print("\n [INFO] Quantization complete \n"
              f"Quantized model with dynamic input shape saved at {quantized_model_path} \n")

        model = onnx.load(quantized_model_path)
        onnx.checker.check_model(model)

        quantized_model_session = InferenceSession(quantized_model_path)

        return quantized_model_session
        
quantizer = SEONNXPTQQuantizer()
sess_quant = quantizer.quantize()