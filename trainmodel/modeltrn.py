import torch
import torch.nn
import torchvision as vision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from PIL import Image
import modelnn

train_path = "../../train/"
test_path = "../../test/"
val_path = "../../val/"


transformers = transforms.Compose([transforms.Resize((64,64),interpolation=Image.NEAREST),\
    transforms.ToTensor(), \
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])])

train_data = vision.datasets.ImageFolder(root=train_path, transform=transformers)
test_data  = vision.datasets.ImageFolder(root=test_path, transform=transformers)
val_data  = vision.datasets.ImageFolder(root=val_path, transform=transformers)

batch_size = 1

train_data_loader = DataLoader(train_data,batch_size=batch_size, shuffle=True)
val_data_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

optimizer = optim.Adam(modelnn.model.parameters(), lr=0.001)
loss_fn  = torch.nn.CrossEntropyLoss()