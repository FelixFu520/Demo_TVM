#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <opencv2/opencv.hpp>
#include <cstdio>
#include <vector>

using namespace std;


bool pairCompare(const std::pair<float, int>& lhs, const std::pair<float, int>& rhs)
{
  return lhs.first > rhs.first;
}

std::vector<int> argmax(const std::vector<float>& v, int topK)
{
	std::vector<std::pair<float, int>> pairs(v.size());
	for (size_t i = 0; i < v.size(); ++i)
		pairs[i] = std::make_pair(v[i], i);

	std::partial_sort(pairs.begin(), pairs.begin() + topK, pairs.end(), pairCompare);

	std::vector<int> result;
	result.reserve(topK);
	for (int i = 0; i < topK; ++i)
		result.push_back(pairs[i].second);
	return result;
}

int main(void) {
  // load the ResNet18 library
  DLDevice dev{kDLCPU, 0};
  tvm::runtime::Module mod_factory = tvm::runtime::Module::LoadFromFile("mod.so");
  
  // create the ResNet18 module
  tvm::runtime::Module resnet18_mod = mod_factory.GetFunction("default")(dev);
  tvm::runtime::PackedFunc set_input = resnet18_mod.GetFunction("set_input");
  tvm::runtime::PackedFunc get_output = resnet18_mod.GetFunction("get_output");
  tvm::runtime::PackedFunc run = resnet18_mod.GetFunction("run");

  // Use the C++ API
  // Replace the input size and data type according to your ResNet18 model
  tvm::runtime::NDArray input = tvm::runtime::NDArray::Empty({32, 3, 224, 224}, DLDataType{kDLFloat, 32, 1}, dev);
  tvm::runtime::NDArray output = tvm::runtime::NDArray::Empty({32, 1000}, DLDataType{kDLFloat, 32, 1}, dev);

  // Set input data (replace with your input data preparation logic)
  // read image
  cv::Mat image = cv::imread("cat.jpg");
  cv::resize(image, image, cv::Size(224, 224));
  cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
  vector<float> mean{0.485, 0.456, 0.406}, std{0.229, 0.224, 0.225};
  // copy
  float* input_data = static_cast<float*>(input->data);
  std::vector<cv::Mat> channels(3);
  cv::split(image, channels);
  for(int b = 0; b<32;b++){
    for(int i=0;i<3;i++){
      for(int j=0;j<224;j++){
        for(int k=0;k<224;k++){
          float point = static_cast<float>(channels[i].at<uchar>(j, k));
          input_data[b*3*224*224 + i*224*224 + j*224 + k] = (point / 255.0 - mean[i]) / std[i];
        }
      }
    }
  }

  // Set the input
  set_input("input", input);
  
  // Run the ResNet18 model
  run();

  // Get the output
  get_output(0, output);

  // 打印Tips
  printf("******Tips:\n");
  string tips = string("280 n02120505 狐狸, grey fox, gray fox, Urocyon cinereoargenteus\n")+
                "281 n02123045 猫, tabby, tabby cat\n"+
                "282 n02123159 猫, tiger cat\n"+
                "283 n02123394 猫, Persian cat\n"+
                "284 n02123597 猫, Siamese cat, Siamese\n"+
                "285 n02124075 猫, Egyptian cat\n"+
                "286 n02125311 猫, cougar, puma, catamount, mountain lion, painter, panther, Felis concolor\n"+
                "287 n02127052 猫, lynx, catamount\n"+
                "288 n02128385 豹, leopard, Panthera pardus\n"+
                "289 n02128757 豹, snow leopard, ounce, Panthera uncia\n"+
                "290 n02128925 豹, jaguar, panther, Panthera onca, Felis onca\n";
  std::cout<<tips;
  // Add your post-processing logic here
  float* output_data = static_cast<float*>(output->data);
  for(int b = 0; b<32; b++){
    float sum = 0;
    std::vector<float> probs(1000);

    // 累加
    for(int idx=0;idx<1000;idx++){
      probs[idx] = exp(output_data[b * 1000 + idx]);
      sum += probs[idx];
    }
    for(int idx=0;idx<1000;idx++){
      probs[idx] /= sum;
    }

    // 找出最大的前5个
    std::vector<int> topK = argmax(probs, 5);
    if(b == 0 || b==1){
      printf("******Top 5:\n");
      for(int idx=0;idx<5;idx++){
        printf("Batch %d: %d, %f\n", b, topK[idx], probs[topK[idx]]);
      }
    }

  }

  return 0;
}