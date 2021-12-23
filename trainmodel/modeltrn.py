import torch
import torchvision as vision
from torchvision.transforms import transforms
from torch.utils.data import DataLoader


train_path = "../../train"
test_path = "../../test"
val_path = "../../val"

transformers = transforms.Compose([transforms.Resize(64), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])])

train_data = vision.datasets.ImageFolder(root=train_path, transform=transformers)
test_data  = vision.datasets.ImageFolder(root=test_path, transform=transformers)
val_data  = vision.datasets.ImageFolder(root=val_path, transform=transformers)

batch_size = 64

train_data_loader = DataLoader(train_data,batch_size=batch_size, shuffle=True)