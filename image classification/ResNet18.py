from torchvision import models
from PIL import Image
import cv2
import torch
from torchsummary import summary
from torchvision import transforms
transform = transforms.Compose([            
# transforms.Resize(256),                    
# transforms.CenterCrop(224),                
 transforms.ToTensor(),                     
 transforms.Normalize(                      
 mean=[0.485, 0.456, 0.406],                
 std=[0.229, 0.224, 0.225]                  
 )])
with open('imagenet_classes.txt') as f:
labels = [line.strip() for line in f.readlines()]
dir(models)
img = Image.open("camel.jpg")
img_t = transform(img)
batch_t = torch.unsqueeze(img_t, 0)
# First, load the model
resnet = models.resnet18(pretrained=True)
summary(resnet, (3, 224,224))
# Second, put the network in eval mode
resnet.eval()
# Third, carry out model inference
preds = resnet(batch_t)
pred, class_idx = torch.max(preds, dim=1)
print(labels[class_idx])
