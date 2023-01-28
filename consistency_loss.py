
import cv2
import numpy as np
from dataset import LFW

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T


'''
https://github.com/zhaoyuzhi/PyTorch-Sobel/blob/main/pytorch-sobel.py
'''
class Sobel(nn.Module):
    def __init__(self):
        super().__init__()

        Gx = torch.tensor([[0.0, -1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]) # np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        Gy = torch.tensor([[0.0, 0.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]]) # np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        self.Gx = Gx.expand(1, 1, 3, 3)
        self.Gy = Gy.expand(1, 1, 3, 3)
    
    def get_gray(self,x):
        ''' 
        Convert image to its gray one.
        '''
        gray_coeffs = [65.738, 129.057, 25.064]
        convert = x.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
        x_gray = x.mul(convert).sum(dim=1)
        return x_gray.unsqueeze(1)
        

    def forward(self, img, mask):
        if img.shape[1] == 3:
            img = self.get_gray(img)
        if mask.shape[1] == 3:
            mask = self.get_gray(mask)
        
        Ix = F.conv2d(img, self.Gx, padding=1)
        Iy = F.conv2d(img, self.Gy, padding=1)
        Mx = F.conv2d(mask, self.Gx, padding=1)
        My = F.conv2d(mask, self.Gy, padding=1)
        M_mag = torch.sqrt(torch.pow(Mx, 2) + torch.pow(My, 2) + 1e-6)
        numerator = torch.sum(M_mag * (1 - (Ix*Mx + Iy*My)**2))
        deminator = torch.sum(M_mag)
        return numerator/deminator

def sobel_test_pytorch():
    img = cv2.imread('data/lfw_data_imgs/Inga_Hall/Inga_Hall_0001.jpg')
    a = img.shape # (256, 256, 3)
    
    img = (img / 255.0).astype(np.float32)
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    img = sobel(img) # input img: data range [0, 1]; data type torch.float32; data shape [1, 3, 256, 256]
    b = img.shape # torch.Size([1, 1, 256, 256])
    img = (img[0, :, :, :].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    
    c = img.shape # (256, 256, 1)
    cv2.imshow('pytorch sobel', img)
    cv2.waitKey(0)
    exit(-1)

def sobel_test_opencv():
    # Sobel Edge Detection
    Ix = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
    Iy = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis

    Mx = cv2.Sobel(src=mask, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
    My = cv2.Sobel(src=mask, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis

    Mmag = cv2.Sobel(src=mask, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection

    consistency_loss = (Mmag * (1 - (Ix * Mx + Iy * My)**2)) / 255.

if __name__ == "__main__":
    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),

        ]
    )
    dataset = LFW("data/lfw_data_imgs", "data/lfw_hair_masks", mode="train", transform=transform, debug=False)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)
    sobel = Sobel()

    
    
    '''
    x_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    y_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    filters = torch.Tensor(x_kernel)
    >>> filters
    tensor([[-1.,  0.,  1.],
            [-2.,  0.,  2.],
            [-1.,  0.,  1.]])
    
    filters = filters.expand(3,3,3).unsqueeze(0)
    
    img_tensor = torch.Tensor(img)
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
    >>> inputs = img_tensor
    >>> out = F.conv2d(inputs, filters, padding=1)
    cv2.imwrite("sobel_x.png", np.array(out[0].squeeze(0)))

    '''
    
    #### TEST
    for idx, (imgs, masks) in enumerate(train_loader):
        out = sobel(imgs, masks.unsqueeze(1).float())
        # print(out.item())
        
        
        break



