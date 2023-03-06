
from torchvision import datasets, transforms
from Models import fcbformer
import torchvision
from PIL import Image
model = fcbformer.FCBFormer()


tsr  = Image.open("train/images/0011.png")

transform2 = transforms.Compose([transforms.ToTensor(),])
tsr = transform2(tsr)
tsr = tsr.unsqueeze(0)
print(tsr.shape)
out = model(tsr)

print(out.shape)