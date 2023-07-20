import gc
from PIL import Image
from IPython import display
import ipywidgets
panda = Image.open("Giant_Panda_2004-03-2.jpg")
koala = Image.open("unique-animals-australia.jpg")
lion = Image.open("Wildlife_at_Maasai_Mara_(Lion).jpg")
sea_lion = Image.open("1280px-monachus_schauinslandi.jpg")
wall_clock = Image.open("51RxQK7kK0L._SY355_.jpg")
digital_clock = Image.open("583309-Product-0-I-637800179303038345.jpg")
from torchvision.transforms.functional import to_pil_image, pil_to_tensor
panda_int = pil_to_tensor(panda)
koala_int = pil_to_tensor(koala)
lion_int = pil_to_tensor(lion)
sea_lion_int = pil_to_tensor(sea_lion)
wall_clock_int = pil_to_tensor(wall_clock)
digital_clock_int = pil_to_tensor(digital_clock)
panda_int.shape, koala_int.shape, lion_int.shape, sea_lion_int.shape, wall_clock_int.shape, digital_clock_int.shape
from torchvision.models import resnet101, ResNet101_Weights
resnet = resnet101(weights=ResNet101_Weights.DEFAULT, progress=False)
resnet.eval();
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
mobilenet = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT, progress=False)
mobilenet.eval();
preprocess_img = ResNet101_Weights.DEFAULT.transforms()
preprocess_img(panda_int).unsqueeze(dim=0).shape
panda_preds1 = resnet(preprocess_img(panda_int).unsqueeze(dim=0))
koala_preds1 = resnet(preprocess_img(koala_int).unsqueeze(dim=0))
lion_preds1 = resnet(preprocess_img(lion_int).unsqueeze(dim=0))
sea_lion_preds1 = resnet(preprocess_img(sea_lion_int).unsqueeze(dim=0))
wall_clock_preds1 = resnet(preprocess_img(wall_clock_int).unsqueeze(dim=0))
digital_clock_preds1 = resnet(preprocess_img(digital_clock_int).unsqueeze(dim=0))
panda_preds1.shape
preprocess_img = MobileNet_V3_Small_Weights.DEFAULT.transforms()
preprocess_img(panda_int).unsqueeze(dim=0).shape
panda_preds2 = resnet(preprocess_img(panda_int).unsqueeze(dim=0))
koala_preds2 = resnet(preprocess_img(koala_int).unsqueeze(dim=0))
lion_preds2 = resnet(preprocess_img(lion_int).unsqueeze(dim=0))
sea_lion_preds2 = resnet(preprocess_img(sea_lion_int).unsqueeze(dim=0))
wall_clock_preds2 = resnet(preprocess_img(wall_clock_int).unsqueeze(dim=0))
digital_clock_preds2 = resnet(preprocess_img(digital_clock_int).unsqueeze(dim=0))
panda_preds2.shape
cats = MobileNet_V3_Small_Weights.DEFAULT.meta["categories"]
preds2 = []
preds2.append([cats[idx] for idx in panda_preds2.argsort()[0].numpy()][::-1][:3])
preds2.append([cats[idx] for idx in koala_preds2.argsort()[0].numpy()][::-1][:3])
preds2.append([cats[idx] for idx in lion_preds2.argsort()[0].numpy()][::-1][:3])
preds2.append([cats[idx] for idx in sea_lion_preds2.argsort()[0].numpy()][::-1][:3])
preds2.append([cats[idx] for idx in wall_clock_preds2.argsort()[0].numpy()][::-1][:3])
preds2.append([cats[idx] for idx in digital_clock_preds2.argsort()[0].numpy()][::-1][:3])
for pred in preds2:
    print(pred)
    import matplotlib.pyplot as plt
fig = plt.figure(figsize=(20,6))
for i, img in enumerate([panda, koala, lion, sea_lion, wall_clock, digital_clock]):
    ax = fig.add_subplot(2,3,i+1)
    ax.imshow(img)
    ax.set_xticks([],[]); ax.set_yticks([],[]);
    ax.text(0,0, "{}\n".format(preds2[i]))
    
