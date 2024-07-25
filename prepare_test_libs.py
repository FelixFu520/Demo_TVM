"""Script to prepare test_resnet18.so"""
import tvm
import numpy as np
from tvm import te
from tvm import relay
from tvm.relay import testing
import os

def resnet18():
    # Define ResNet18 model
    mod, params = testing.resnet.get_workload(
        num_layers=18, batch_size=1, dtype="float32", image_shape=(3, 224, 224)
    )
    return mod, params

def prepare_resnet18_lib():
    mod, params = resnet18()
    # Compile ResNet18 library as dynamic library for riscv64-linux-gnu
    target = tvm.target.Target( 
        "llvm -device=x86_64 -mtriple=x86_64-linux-gnu -mattr=+avx2" 
    )
  
    with relay.build_config(opt_level=3):
        module = relay.build(mod, target=target, params=params)

    module.export_library("test_resnet18.so", cc="gcc")
    print("save finish")

if __name__ == "__main__":
    prepare_resnet18_lib()