import numpy as np
import random
import image
import torch
from torchvision import transforms, utils

class RandomTranslate(object):
    
    def __call__(self, sample):
        
        img, bb  = sample['image'], sample['bb']
                
        tx = int(np.random.uniform(-50,+50))
        ty = int(np.random.uniform(-50,+50))
        t = (tx,ty)
        
        img.translate(t)
        bb.translate((t[1],t[0]))
        
        return {'image': img, 'bb': bb}

class CentralCrop(object):
    
    def __call__(self, sample):
        
        img, bb  = sample['image'], sample['bb']
        
        bottom_right = [480,400]
        top_left = [160,80]             
        
        img.crop(top_left,bottom_right)
        bb.offset((640,480),(320,320))
        
        return {'image': img, 'bb': bb}

class RandomRotate(object):
    
    def __call__(self, sample):
        
        img, bb  = sample['image'], sample['bb']
        
        #rotations = [0, np.pi / 3, np.pi / 4, np.pi / 6, np.pi / 2, 3 * np.pi / 2]
        #rot = random.choice(rotations)
        
        rot = np.random.uniform(-np.pi/2,np.pi/2)
        
        img.rotate(rot,center = [160,160])
        bb.rotate(rot,center = [160,160])
        
        return {'image': img, 'bb': bb}
    
class Rescale(object):
    
    def __call__(self, sample):
        
        img, bb  = sample['image'], sample['bb']
        
        bb.rescale((320,320),(224,224))
        img.resize((224,224))
        
        return {'image': img, 'bb': bb}
        
class Normalize(object):
    
    def __call__(self, sample):        
        img, bb  = sample['image'], sample['bb']
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])        
        img = normalize(img)
        return {'image': img, 'bb': bb}

class ToTensor(object):
      
    def __call__(self, sample):
        img, bb  = sample['image'], sample['bb']
        img = img.astype(np.float32) / 255.0
        img = img.transpose((2, 0, 1))
        coordinate = np.array([bb.center[0],bb.center[1],np.cos(bb.angle),np.sin(bb.angle),bb.length,bb.width])
        
        img = torch.from_numpy(img)
        coordinate = torch.from_numpy(coordinate)
        
        return {'image': img,'bb': coordinate}

