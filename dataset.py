import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torchvision.transforms as T



class LFW(torch.utils.data.Dataset):
    def __init__(self, root_imgs, root_masks, mode="train", transform=None, debug=False):
        self.root_imgs = root_imgs
        self.root_masks = root_masks
        self.transform = transform
        self.mode = mode
        self.mode_img_paths = []

        self.max_number_rigid = 4

        self.debug = debug
        
        if mode == "train":
            self.split_file_path = "data/parts_train.txt"
        elif mode == "validation":
            self.split_file_path = "data/parts_validation.txt"
        elif mode == "test":
            self.split_file_path = "data/parts_test.txt"
        else:
            assert "Unsported mode"
            exit(-1)
        
        self.create_mode_paths()
        

        self.mode_mask_paths = [os.path.join(self.root_masks, (os.path.split(img_path)[-1].split(".")[0] + ".ppm"))  for img_path in self.mode_img_paths]
    
    def create_mode_paths(self):
        with open(self.split_file_path, "r") as f:
            lines = f.readlines()
        for line in lines:
            if self.debug:
                assert len(line.rsplit()) == 2, "There is more than 2 items in one line!"
            name, number = line.rsplit()
            sub_path = os.path.join(self.root_imgs, name)
            out = os.path.join(sub_path, (name + "_" + (self.max_number_rigid - len(str(int(number)))) * "0" + str(int(number)) + ".jpg"))
            self.mode_img_paths.append(out)

    
    def __len__(self):
        return len(self.mode_mask_paths)

    
    def __getitem__(self, index):
        img_path = self.mode_img_paths[index]
        mask_path = self.mode_mask_paths[index]
        if self.debug:
            print(f" img name {img_path}")
            print(f" mask name {mask_path}")
        
        img = Image.open(img_path).convert("RGB")
        mask = cv2.imread(mask_path, cv2.COLOR_BGR2RGB)
        # mask = Image.open(mask_path).convert("RGB")
        
        hair = (np.array(mask) == [255, 0, 0]).all(-1)
        face = (np.array(mask) == [0, 255, 0]).all(-1)
        background = (np.array(mask) == [0, 0, 255]).all(-1)

        bw_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=int)
        bw_mask[background] = [1]
        bw_mask[hair] = [0]
        # mask = tf.one_hot(empty_masked, self.NUM_CLASS, dtype=tf.int32) # 512, 512, NUM_CLASS

        bw_mask = torch.from_numpy(bw_mask)
        if self.transform:
            img = self.transform(img)
            # bw_mask = self.transform(bw_mask)
                
        return img, bw_mask


if __name__ == "__main__":
    # img_test = cv2.imread("data/lfw_hair_masks/Inga_Hall_0001.ppm", cv2.COLOR_BGR2RGB)
    # cv2.imshow("test_img", img_test)
    # cv2.waitKey(0)
    # img_test = Image.open("data/lfw_hair_masks/Inga_Hall_0001.ppm") # .convert("RGB")
    
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
        print(imgs.shape)
        print(masks.shape)
        cv2.imwrite("test_mask.png", np.array(masks[0]))
        exit(-1)
        img = cv2.cvtColor(np.array(imgs[0].permute(1, 2, 0)), cv2.COLOR_RGB2BGR)
        mask = np.array(masks[0])
        print(f"mask shape {mask.shape}")
        exit(-1)
        cv2.imshow("img", img)
        cv2.waitKey(0)
        plt.imshow(mask, cmap="gray")
        plt.show()
        # cv2.imshow("mask", mask)
        # cv2.waitKey(0)
        
        break
