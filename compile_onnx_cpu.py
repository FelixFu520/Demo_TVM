# 参考：https://tvm.apache.org/docs/how_to/compile_models/from_onnx.html?highlight=onnx

import onnx
import numpy as np
import tvm
from tvm import te
import tvm.relay as relay
import cv2


######################################################################
# Load pretrained ONNX model
# ---------------------------------------------
onnx_model = onnx.load("resnet18.onnx")


######################################################################
# Load a test image
# ---------------------------------------------
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
image = cv2.imread("cat.jpg")
image = cv2.resize(image, (224, 224))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image / 255.0
image = image - mean
image = image / std
image = image.transpose((2, 0, 1))
image_batch = np.expand_dims(image, axis=0).astype(np.float32)
x = np.repeat(image_batch, 32, axis=0)


######################################################################
# Compile the model with relay
# ---------------------------------------------
# Typically ONNX models mix model input values with parameter values, with
# the input having the name `input`. This model dependent, and you should check
# with the documentation for your model to determine the full input and
# parameter name space.
#
# Passing in the shape dictionary to the `relay.frontend.from_onnx` method
# tells relay which ONNX parameters are inputs, and which are parameters, and
# provides a static definition of the input size.
target = "llvm"

input_name = "input"
shape_dict = {input_name: x.shape}
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
# Compile ResNet18 library as dynamic library for riscv64-linux-gnu
target = tvm.target.Target( 
    "llvm -device=x86_64 -mtriple=x86_64-linux-gnu -mattr=+avx2" 
)

with relay.build_config(opt_level=3):
    module = relay.build(mod, target=target, params=params)

module.export_library("mod.so", cc="gcc")
print("save finish")

with tvm.transform.PassContext(opt_level=1):
    executor = relay.build_module.create_executor(
        "graph", mod, tvm.cpu(0), target, params
    ).evaluate()


######################################################################
# Execute on TVM
# ---------------------------------------------
dtype = "float32"
tvm_output = executor(tvm.nd.array(x.astype(dtype))).numpy()
print(tvm_output.shape)
print("compile finish")

######################################################################
# Notes
# ---------------------------------------------
# By default, ONNX defines models in terms of dynamic shapes. The ONNX importer
# retains that dynamism upon import, and the compiler attempts to convert the model
# into a static shapes at compile time. If this fails, there may still be dynamic
# operations in the model. Not all TVM kernels currently support dynamic shapes,
# please file an issue on discuss.tvm.apache.org if you hit an error with dynamic kernels.
#
# This particular model was build using an older version of ONNX. During the import
# phase ONNX importer will run the ONNX verifier, which may throw a `Mismatched attribute type`
# warning. Because TVM supports a number of different ONNX versions, the Relay model
# will still be valid.