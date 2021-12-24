import torch
from torchvision.transforms import transforms
from modelnn import Net
from PIL import Image
import modeltrn

labels = ['cat','dog']

loaded_model = Net()
model_dict = torch.load('model/net/final.model')
loaded_model.load_state_dict(model_dict)

img = Image.open('test.jpg')
img = modeltrn.transformers(img)
img = img.unsqueeze(0)


predict = loaded_model(img)
print(predict)
prediction = predict.argmax()
print(prediction)
print(labels[prediction])
