// Description: A simple demo to show how to use TVM runtime to load a model and run it on CPU.
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>
#include <dmlc/memory_io.h>
#include <opencv2/opencv.hpp>

#include <cstdio>
#include <fstream>

using namespace std;
using namespace cv;
using namespace tvm;
using namespace tvm::runtime;



int main(int argc, char** argv) {
    cout << "Hello, TVM(CPU)!" << endl;

    // load in library
    DLDevice dev{kDLCPU, 0};
    tvm::runtime::Module mod_factory = tvm::runtime::Module::LoadFromFile("resnet18-cpu/mod.so");

    // create graph executor modulec
    tvm::runtime::Module gmod = mod_factory.GetFunction("default")(dev);
    tvm::runtime::PackedFunc set_input = gmod.GetFunction("set_input");
    tvm::runtime::PackedFunc get_output = gmod.GetFunction("get_output");
    tvm::runtime::PackedFunc run = gmod.GetFunction("run");

    // read image
    Mat image = imread("cat.jpg");
    resize(image, image, Size(224, 224));
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    image.convertTo(image, CV_32FC3);
    image = image / 255.0;
    cv::Scalar mean{0.485, 0.456, 0.406}, std{0.229, 0.224, 0.225};
    meanStdDev(image, mean, std);


//     mean = [0.485, 0.456, 0.406]
// std = [0.229, 0.224, 0.225]
// image = cv2.imread("cat.jpg")
// image = cv2.resize(image, (224, 224))
// image = image / 255.0
// image = image - mean
// image = image / std
// image = image.transpose((2, 0, 1))
// image_batch = np.expand_dims(image, axis=0).astype(np.float32)
// image_batch = np.repeat(image_batch, 32, axis=0)
// image_batch_tvm = tvm.nd.array(image_batch, ctx)
    
    return 0;
}
