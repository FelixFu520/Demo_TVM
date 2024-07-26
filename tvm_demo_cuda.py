from tvm.contrib import graph_executor
from tvm.driver.tvmc.model import TVMCPackage
from tvm.driver.tvmc.runner import get_input_info, make_inputs_dict
from tvm.driver import tvmc
import tvm.runtime
from tvm import relay
from tvm import rpc
import tvm
import time
from PIL import Image
import numpy as np
import cv2




# 准备模型
ctx = tvm.cuda(0)
tvmc_package = TVMCPackage(package_path="resnet18-cuda.tar")
lib = tvm.runtime.load_module(tvmc_package.lib_path)
m = graph_executor.create(tvmc_package.graph, lib, ctx)
m.load_params(tvmc_package.params)
print("model loaded")


# 准备数据
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
image_batch = np.repeat(image_batch, 32, axis=0)
image_batch_tvm = tvm.nd.array(image_batch, ctx)


# 推理
shape_dict, dtype_dict = get_input_info(tvmc_package.graph, tvmc_package.params)
print(f"shape_dict:{shape_dict}")
print(f"dtype_dict:{dtype_dict}")
m.set_input("input", image_batch_tvm)
m.run()

# 后处理
output = m.get_output(0).asnumpy()
class_cat = output[0]
softmax_cat = np.exp(class_cat) / np.sum(np.exp(class_cat))
index_of_cat = np.argmax(softmax_cat)
tips =  """
TIPS:
280 n02120505 狐狸, grey fox, gray fox, Urocyon cinereoargenteus
281 n02123045 猫, tabby, tabby cat
282 n02123159 猫, tiger cat
283 n02123394 猫, Persian cat
284 n02123597 猫, Siamese cat, Siamese
285 n02124075 猫, Egyptian cat
286 n02125311 猫, cougar, puma, catamount, mountain lion, painter, panther, Felis concolor
287 n02127052 猫, lynx, catamount
288 n02128385 豹, leopard, Panthera pardus
289 n02128757 豹, snow leopard, ounce, Panthera uncia
290 n02128925 豹, jaguar, panther, Panthera onca, Felis onca"""
print(f"cat.jpg: {index_of_cat}")
print(f"cat.jpg score:{softmax_cat[index_of_cat]}")


# 耗时测试
t0 = time.time()
for i in range(100):
    m.set_input("input", image_batch_tvm)
    m.run()
c = time.time() - t0
print(f"infer 100 cost average: {c/100}")
