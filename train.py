from Models import fcbformer
from Data import datacfg
import torchvision
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch

device = 'cuda'

transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),])
transform2 = transforms.Compose([transforms.ToTensor(),])

dataset = datacfg.UlcerData('images', 'labels',transform, transform2)

train_loader = DataLoader(dataset,batch_size=16,shuffle = True)

model = fcbformer.FCBFormer()


lossF = nn.BCEWithLogitsLoss() 

optimizer = torch.optim.AdamW([ 
    dict(params=model.parameters(), lr=0.001),
])

model.to(device)


for epoch in range(80): 
    for i, data in enumerate(train_loader):

        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = lossF(outputs, labels)
        loss.backward()
        optimizer.step()
               
        
            
        print(f"Epoch {epoch} i {i}, Training loss {loss.item():.4f},")
print('Finished Training')