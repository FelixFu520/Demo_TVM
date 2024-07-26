# TVM DEMO
基于tvm 0.18

## 安装依赖环境
`pip install -r requirements.txt`

## 安装tvm
参考: https://siwrc302o4r.feishu.cn/wiki/OB63wVvo1ih0iukOVSWcKDJwneg
安装完毕后, 将tvm复制到third_party目录下。 记得安装tvm的python包

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
### 使用TVM(CPU)，C++推理
```
apt-get install libopencv-dev
tvmc compile --target llvm resnet18.onnx -o resnet18-cpu.tar # 这种编译出来模型在C++中用不了, 应该是参数设置问题, 待解决
python compile_onnx_cpu.py  # 改变模型编译方式


```
### 使用TVM(CUDA)，C++推理
```
tar -xvf resnet18-cuda.tar


```

## 参考
- https://github.com/YiyaoYang1/tvm-riscv-deploy
- https://tvm.apache.org/docs/how_to/compile_models/from_onnx.html?highlight=onnx