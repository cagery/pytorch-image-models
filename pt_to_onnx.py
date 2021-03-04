#!/usr/bin/python3

import timm
from timm import list_models, create_model, set_scriptable

import io
import os
import numpy as np
from torch import nn
import torch.utils.model_zoo as model_zoo
import torch.onnx
from PIL import Image
import urllib.request

import onnx
import onnxruntime


def parseToOnnx():
    model = create_model("dm_nfnet_f0", pretrained=True)
    model.eval()

    input_size = model.default_cfg['input_size']
    batch_size = 1

    inputs = torch.randn((batch_size, *input_size))
    outputs = model(inputs)
    assert outputs.shape[0] == batch_size
    assert not torch.isnan(outputs).any(), 'Output included NaNs'


    torch.onnx.export(model,               # model being run
                      inputs,                         # model input (or a tuple for multiple inputs)
                      "dm_nfnet_f0.onnx",   # where to save the model (can be a file or file-like   object)
                      export_params=True,        # store the trained parameter weights inside the model     file
                      opset_version=11,          # the ONNX version to export the model to
                      do_constant_folding=False,  # whether to execute constant folding for optimization
                      input_names = ['inputs'],   # the model's input names
                      output_names = ['outputs'], # the model's output names
                      dynamic_axes={'inputs' : {0 : 'batch_size'},    # variable lenght axes
                                    'outputs' : {0 : 'batch_size'}})

def testOnnxFile():
    onnx_model = onnx.load("dm_nfnet_f0.onnx")
    onnx.checker.check_model(onnx_model)

testOnnxFile()

# Run PyTorch model
model = create_model("dm_nfnet_f0", pretrained=True)
model.eval()

input_size = model.default_cfg['input_size']
batch_size = 1
inputs = torch.randn((batch_size, *input_size))
outputs = model(inputs)
assert outputs.shape[0] == batch_size
assert not torch.isnan(outputs).any(), 'Output included NaNs'


# Compare ONNX runtime to PyTorch
ort_session = onnxruntime.InferenceSession("dm_nfnet_f0.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(inputs)}
ort_outs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(outputs), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")