from CornellDataset import CornellDataset
import image
import grasp
import numpy as np
import matplotlib.pyplot as plt
from transformation import RandomTranslate,CentralCrop,RandomRotate,Rescale,Normalize,ToTensor
from torchvision import transforms, utils
import random


path = 'Dataset'

dataset = CornellDataset(path,transform=transforms.Compose([RandomTranslate(),
                                                            CentralCrop(),
                                                            RandomRotate(),
                                                            Rescale(),
                                                            ToTensor(),
                                                            Normalize()                                                            
                                                            ]))

for i in range(2):
    np.random.seed(20)
    random.seed(20)                                                        
    sample = dataset.__getitem__(1)
    
    img = sample['image']
    bb =  sample['bb']
    
    print(img)
    im = img.numpy()
    im = im.transpose((1,2,0))
    img = image.Image(im)
    
    gtt = grasp.Grasp((bb[0],bb[1]),np.arctan(bb[3]/bb[2]),bb[4],bb[5])
    bb = gtt.as_gr
    
    figure, ax = plt.subplots(nrows=1, ncols=1)
    img.show(ax)
    bb.plot(ax)

