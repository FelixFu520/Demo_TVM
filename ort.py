import cv2
import onnxruntime as ort
import numpy as np
import time

# 准备模型
ort_session = ort.InferenceSession("resnet18.onnx")


# 准备数据
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
image = cv2.imread("cat.jpg")
image = cv2.resize(image, (224, 224))
image = image / 255.0
image = image - mean
image = image / std
image = image.transpose((2, 0, 1))
image_batch = np.expand_dims(image, axis=0).astype(np.float32)
image_batch = np.repeat(image_batch, 32, axis=0)


# 推理
output = ort_session.run(
    None,
    {"input": image_batch}
)[0]


# 后处理
class_cat = output[0]
softmax_cat = np.exp(class_cat) / np.sum(np.exp(class_cat))
index_of_cat = np.argmax(softmax_cat)
print("285 n02124075 猫, Egyptian cat")
print(f"cat.jpg: {index_of_cat}")
print(f"cat.jpg score:{softmax_cat[index_of_cat]}")


# 耗时测试
t0 = time.time()
for i in range(100):
    ort_session.run(
        None,
        {"input": image_batch}
    )
c = time.time() - t0
print(f"infer 100 cost average: {c/100}")


# output
# 285 n02124075 猫, Egyptian cat
# cat.jpg: 285
# cat.jpg score:0.48572447896003723
# infer 100 cost average: 0.061029350757598876