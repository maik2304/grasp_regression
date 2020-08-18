import glob
import os
import torch
import grasp, image
import numpy as np
import random
import matplotlib.pyplot as plt


class CornellDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper for the Cornell dataset.
    """

    def __init__(self, file_path,output_size=224,transform = None):
        """
        :param file_path: Cornell Dataset directory.
        
        """
        
        self.output_size = output_size
        
        self.grasp_files = glob.glob(os.path.join(file_path,'pcd*cpos.txt'))
        self.grasp_files.sort()
        self.rgb_files = [f.replace('cpos.txt', 'r.png') for f in self.grasp_files]
        
        self.data_len = len(self.grasp_files)
        self.len = 1*self.data_len
        
        self.transform = transform
        
        if  self.data_len == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(file_path))
        
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        
        idx = index % self.data_len
        
        img = image.Image.from_file(self.rgb_files[idx])
        bbs = grasp.GraspRectangles.load_from_cornell_file(self.grasp_files[idx])
        c = random.choice(list(range(bbs.length)))
        bb=bbs.grs[c] 
        
        sample =  {'image': img, 'bb': bb}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample