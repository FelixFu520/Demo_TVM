import torch
import torchvision


x = torch.randn(32, 3, 224, 224, device="cuda")
model = torchvision.models.resnet18(pretrained=True).cuda()

torch.onnx.export(
    model,
    x,
    "resnet18.onnx",
    verbose=True,
    input_names=["input"],
    output_names=["output"]
)


