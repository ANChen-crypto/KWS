import torch
import numpy as np
import torch.nn as nn
from modeling import ConformerForCTCKWSOnnx, ConformerForKWSOnnx, LSTMQ, CNNOnnx


encoder_dim = 64
feed_forward_expansion_factor = 4

# model = ConformerForCTCKWSOnnx(
#     input_dim = 35,
#     encoder_dim = encoder_dim,
#     num_layers = 3,
#     num_attention_heads = 4,
#     feed_forward_expansion_factor = feed_forward_expansion_factor,
#     conv_expansion_factor = 2,
#     input_dropout_p = 0.1,
#     feed_forward_dropout_p = 0.1,
#     attention_dropout_p = 0.1,
#     conv_dropout_p = 0.1,
#     conv_kernel_size = 31,
#     half_step_residual = True,
# )

model = CNNOnnx()

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

ckpt = torch.load(r"weights\cnn_ctc_0.1124.pth", weights_only=True, map_location="cpu")
print("best val loss:", ckpt["best_val_loss"].item())
model.load_state_dict(ckpt["model"], strict=False)

model.eval()

dummy_input = torch.randn(1, 35, 197)
dummy_input_pos = torch.randn(1, 21, encoder_dim)

# torch.onnx.export(model,
#                   (dummy_input, dummy_input_pos),
#                   "q_weights/model_ctc.onnx",
#                   export_params=True,
#                   opset_version=17,
#                   do_constant_folding=True,
#                   input_names = ['inputs', 'inputs_pos'],
#                   output_names = ['output'],
#                   dynamic_axes={'inputs' : {0: 'batch_size'}, 'inputs_pos' : {0: 'batch_size'}, 'output': {0 : 'batch_size'}}
#                 )

torch.onnx.export(model,
                  (dummy_input),
                  "q_weights/model_ctc.onnx",
                  export_params=True,
                  opset_version=17,
                  do_constant_folding=True,
                  input_names = ['inputs'],
                  output_names = ['output'],
                  dynamic_axes={'inputs' : {0: 'batch_size'}, 'output': {0 : 'batch_size'}}
                )
