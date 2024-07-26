# TVM DEMO
各个库版本说明
```
cmake --version
cmake version 3.30.1

gcc -v
gcc version 9.4.0 (Ubuntu 9.4.0-1ubuntu1~20.04.2)

make -v
GNU Make 4.2.1

python -V
Python 3.8.10

tvmc --version
0.18.dev0

pkg-config --modversion zlib
1.3.1

nvcc -V
Cuda compilation tools, release 11.3, V11.3.109

find / -name libnccl
root@3a867420f6ac:~/data/demo_tvm# dpkg -l | grep nccl
hi  libnccl-dev                          2.9.9-1+cuda11.3                      amd64        NVIDIA Collective Communication Library (NCCL) Development Files
hi  libnccl2                             2.9.9-1+cuda11.3                      amd64        NVIDIA Collective Communication Library (NCCL) Runtime

python -c "import torch; print(torch.__version__)"
1.12.1+cu113

python -c "import torch; print(torch.cuda.is_available())"
True


```
## 使用Docker
```
docker build -t 镜像名称 . -f dockerfile
docker run -p 主机端口号:22 --name 容器名称 -itd -v /data:/root/data --gpus all --privileged --shm-size=64g 镜像名称
```
下面所有操作都是在容器中进行的
## 安装依赖环境

```
pip install -r requirements.txt
```

## 安装tvm
参考: https://siwrc302o4r.feishu.cn/wiki/OB63wVvo1ih0iukOVSWcKDJwneg
安装完毕后, 将tvm复制到third_party目录下。 记得安装tvm的python包
## 安装TensorRT
为了对比速度, 装了下TensorRT(8.6.0.12), 下载[安装包](https://pan.baidu.com/s/1l72iuoL74s_omZ1jCa18kQ?pwd=054e), 放到third_party目录下

```
tar -xvf TensorRT-8.6.0.12.Linux.x86_64-gnu.cuda-11.8.tar.gz

```
## 运行
### 获取onnx模型
```
python pth2onnx.py
```
### 使用onnxruntime(CPU)推理
```
python ort.py

cat.jpg: 281
cat.jpg score:0.5740500688552856
infer 100 cost average: 0.05617419958114624
```
### 使用tvm(CPU)推理
```
tvmc compile --target llvm resnet18.onnx -o resnet18-cpu.tar
python tvm_demo_cpu.py

cat.jpg: 281
cat.jpg score:0.5740497708320618
infer 100 cost average: 0.4064192843437195
```
### 使用tvm(GPU)推理
```
tvmc compile --target cuda resnet18.onnx -o resnet18-cuda.tar
python tvm_demo_gpu.py

cat.jpg: 281
cat.jpg score:0.5740503668785095
infer 100 cost average: 0.01991621971130371
```
### 使用TVM(CPU/CUDA), TensorRT(CUDA), C++推理
```
apt-get install libopencv-dev
tvmc compile --target llvm resnet18.onnx -o resnet18-cpu.tar # 这种编译出来模型在C++中用不了, 应该是参数设置问题, 待解决
python compile_onnx_cpu.py  # 改变模型编译方式
python_compile_onnx_cuda.py
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE="Release" ..
make
./demo_tvm_cpu
******Top 5:
Batch 0: 281, 0.574050
Batch 0: 285, 0.174399
Batch 0: 282, 0.168115
Batch 0: 287, 0.015327
Batch 0: 728, 0.013453
******Top 5:
Batch 1: 281, 0.574050
Batch 1: 285, 0.174399
Batch 1: 282, 0.168115
Batch 1: 287, 0.015327
Batch 1: 728, 0.013453
infer 100 cost time: 0.198359

./demo_tvm_cuda
******Top 5:
Batch 0: 281, 0.574050
Batch 0: 285, 0.174398
Batch 0: 282, 0.168116
Batch 0: 287, 0.015327
Batch 0: 728, 0.013453
******Top 5:
Batch 1: 281, 0.574050
Batch 1: 285, 0.174398
Batch 1: 282, 0.168116
Batch 1: 287, 0.015327
Batch 1: 728, 0.013453
infer 100 cost time: 0.019427

./demo_trt
******Top 5:
Batch 0: 281, 0.573568
Batch 0: 285, 0.174928
Batch 0: 282, 0.168227
Batch 0: 287, 0.015526
Batch 0: 728, 0.013489
******Top 5:
Batch 1: 281, 0.573568
Batch 1: 285, 0.174928
Batch 1: 282, 0.168227
Batch 1: 287, 0.015526
Batch 1: 728, 0.013489
infer 100 cost time: 0.002790
```

## 参考
- https://github.com/YiyaoYang1/tvm-riscv-deploy
- https://tvm.apache.org/docs/how_to/compile_models/from_onnx.html?highlight=onnx