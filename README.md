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

285 n02124075 猫, Egyptian cat
cat.jpg: 285
cat.jpg score:0.48572447896003723
infer 100 cost average: 0.057646191120147704
```
### 使用tvm(CPU)推理
```
tvmc compile --target llvm resnet18.onnx -o resnet18-cpu.tar
python tvm_demo_cpu.py

model loaded
shape_dict:{'input': [32, 3, 224, 224]}
dtype_dict:{'input': 'float32'}
285 n02124075 猫, Egyptian cat
cat.jpg: 285
cat.jpg score:0.4857271611690521
infer 100 cost average: 0.39994378328323366
```
### 使用tvm(GPU)推理
```
tvmc compile --target cuda resnet18.onnx -o resnet18-cuda.tar
python tvm_demo_gpu.py

model loaded
shape_dict:{'input': [32, 3, 224, 224]}
dtype_dict:{'input': 'float32'}
285 n02124075 猫, Egyptian cat
cat.jpg: 285
cat.jpg score:0.48572424054145813
infer 100 cost average: 0.019815213680267334

```
### 使用TVM(CPU)，C++推理
```

```