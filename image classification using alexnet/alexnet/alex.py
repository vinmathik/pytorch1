from torchvision import models
import torch
dir(models)
alexnet = models.alexnet(pretrained=True)
print(alexnet)
from torchvision import transforms
transform = transforms.Compose
([            
 transforms.Resize(256),                    
 transforms.CenterCrop(224),                
 transforms.ToTensor(),                     
 transforms.Normalize(                      
 mean=[0.485, 0.456, 0.406],                
 std=[0.229, 0.224, 0.225]                  
 )])
from PIL import Image
img = Image.open("dog.jpg")
img_t = transform(img)
batch_t=torch.unsqueeze(img_t,0)
print(batch_t)
alexnet.eval()
out = alexnet(batch_t)
print(out.shape)
with open('imagenet_classes.txt') as f:
 classes = [line.strip() for line in f.readlines()]
 _, indices = torch.sort(out, descending=True)
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
[(classes[idx], percentage[idx].item()) for idx in indices[0][:5]]
[('Labrador retriever', 41.585166931152344),
 ('golden retriever', 16.59166145324707),
 ('Saluki, gazelle hound', 16.286880493164062),
 ('whippet', 2.8539133071899414),
 ('Ibizan hound, Ibizan Podenco', 2.3924720287323)]
