#include <opencv2/opencv.hpp>
#include <cstdio>
#include <vector>
#include <filesystem>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>


using namespace std;
using namespace cv;
namespace fs = std::filesystem;
using namespace nvinfer1;

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

class NVLogger : public nvinfer1::ILogger {
public:
	nvinfer1::ILogger::Severity reportableSeverity;

	explicit NVLogger(nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kINFO) :
		reportableSeverity(severity)
	{
	}

	void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override{
    if (severity > reportableSeverity) {
      return;
    }
    switch (severity) {
    case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
        cout<<"error:" << msg << endl;
        break;
    case nvinfer1::ILogger::Severity::kERROR:
        cout<<"error:" << msg << endl;
        break;
    case nvinfer1::ILogger::Severity::kWARNING:
        cout<<"warning:" << msg << endl;
        break;
    case nvinfer1::ILogger::Severity::kINFO:
        cout<<"info:" << msg << endl;
        break;
    default:
        break;
    }
  }
};

int main(void) {
  std::shared_ptr<NVLogger> m_logger{ new NVLogger() };

  // 转换模型
  if(!fs::exists("../resnet18.trt")){
    cout << "building ...." << endl;

    auto builder = unique_ptr<IBuilder>(createInferBuilder(*m_logger));
    if (builder == nullptr) {
      return 1;
    }

    auto network = unique_ptr<INetworkDefinition>(builder->createNetworkV2(1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));
    if (network == nullptr) {
      return 1;
    }

    auto parser = unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, *m_logger));
    if (parser == nullptr) {
      return 1;
    }

    if (!parser->parseFromFile(string("../resnet18.onnx").c_str(), static_cast<int>(ILogger::Severity::kWARNING))) {
      return 1;
    }

    auto config = unique_ptr<IBuilderConfig>(builder->createBuilderConfig());
    if (config == nullptr) {
      return 1;
    }

    config->setFlag(BuilderFlag::kFP16);

    auto plan = unique_ptr<IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    if (plan == nullptr) {
      return 1;
    }

    auto runtime = unique_ptr<IRuntime>(createInferRuntime(*m_logger));
    if (runtime == nullptr) {
      return 1;
    }

    auto engine = unique_ptr<ICudaEngine>(runtime->deserializeCudaEngine(plan->data(), plan->size()));
    if (engine == nullptr) {
      return 1;
    }

    auto serialized = unique_ptr<IHostMemory>(engine->serialize());
    if (serialized == nullptr) {
      return 1;
    }
    ofstream p(string("../resnet18.trt").c_str(), ios::binary);
    p.write(reinterpret_cast<const char*>(serialized->data()), serialized->size());
    p.close();
  }

  // 读模型文件
  string file_path = string("../resnet18.trt");
  fstream file;
  file.open(file_path.c_str(), ios::in | ios::binary);
  if (!file.is_open()) {
    return 1;
  }
  file.seekg(0, std::ios::end);
  int length = file.tellg();
  file.seekg(0, std::ios::beg);
  std::unique_ptr<char[]> data(new char[length]);
  file.read(data.get(), length);
  file.close();

  // 生成Runtime
  std::shared_ptr<nvinfer1::IRuntime> runtime = shared_ptr<IRuntime>(createInferRuntime(*m_logger));
  if (runtime == nullptr) {
    return 1;
  }

  // 加载Engine
  std::shared_ptr<nvinfer1::ICudaEngine> engine = shared_ptr<ICudaEngine>(runtime->deserializeCudaEngine(data.get(), length));
  if (engine == nullptr) {
    return 1;
  }

  // 生成Context
  std::shared_ptr<nvinfer1::IExecutionContext> context = shared_ptr<IExecutionContext>(engine->createExecutionContext());
  if (context == nullptr) {
    return 1;
  }

  // 读取图片
  cv::Mat image = cv::imread("cat.jpg");
  cv::resize(image, image, cv::Size(224, 224));
  cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
  vector<float> mean{0.485, 0.456, 0.406}, std{0.229, 0.224, 0.225};
  // copy
  vector<float> input_data(32*3*224*224);
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

  // 分配内存
  vector<void*> bindings(2);
  vector<void*> output(1);
  cudaMalloc(&bindings[0], 32*3*224*224*sizeof(float));
  cudaMalloc(&bindings[1], 32*1000*sizeof(float));
  output[0] = malloc(32*1000*sizeof(float));

  // 拷贝数据
  cudaMemcpy(bindings[0], input_data.data(), 32*3*224*224*sizeof(float), cudaMemcpyHostToDevice);

  // 推理
  context->executeV2(bindings.data());

  // 拷贝结果
  cudaMemcpy(output[0], bindings[1], 32*1000*sizeof(float), cudaMemcpyDeviceToHost);

  // 后处理
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
  float* output_data = static_cast<float*>(output[0]);
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

  // 统计时间
  auto start = cv::getTickCount();
  for(int i=0;i<100;i++){
    context->executeV2(bindings.data());
  }
  auto end = cv::getTickCount();
  double time = (end - start) / cv::getTickFrequency() / 100;
  printf("infer 100 cost time: %f\n", time);

  // 释放显存和内存
  for (int t = 0; t < bindings.size(); t++) {
    cudaFree(bindings[t]);
  }
  for (int t = 0; t < output.size(); t++) {
    free(output[t]);
  }

  
  return 0;
}