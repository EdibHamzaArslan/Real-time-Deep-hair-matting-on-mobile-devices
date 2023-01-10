
import cv2
import numpy as np
from dataset import LFW

import torch
import torchvision.transforms as T



if __name__ == "__main__":
    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
        ]
    )
    dataset = LFW("data/lfw_data_imgs", "data/lfw_hair_masks", mode="train", transform=transform, debug=False)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)
    
    #### TEST
    for idx, (imgs, masks) in enumerate(train_loader):
        img = np.array(imgs[0].permute(1, 2, 0))
        mask = np.array(masks[0].permute(1, 2, 0))
        cv2.imshow("img", img) 
        cv2.imshow("mask", mask)
        # Sobel Edge Detection
        Ix = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
        Iy = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis

        Mx = cv2.Sobel(src=mask, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
        My = cv2.Sobel(src=mask, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis

        Mmag = cv2.Sobel(src=mask, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection

        consistency_loss = (Mmag * (1 - (Ix * Mx + Iy * My)**2)) / 255.
        print(f"consistency loss shape {consistency_loss.shape}")
        print(consistency_loss)
        cv2.imshow("loss", consistency_loss)
        cv2.waitKey(0)

        
        # img = cv2.cvtColor(np.array(imgs[0].permute(1, 2, 0)), cv2.COLOR_RGB2BGR)
        # mask = cv2.cvtColor(np.array(masks[0].permute(1, 2, 0)), cv2.COLOR_RGB2BGR)
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        # cv2.imshow("mask", mask)
        # cv2.waitKey(0)
        
        break



