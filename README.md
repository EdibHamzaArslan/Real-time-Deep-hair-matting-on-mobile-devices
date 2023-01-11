# Real-time-Deep-hair-matting-on-mobile-devices
https://arxiv.org/abs/1712.07168


**<h3>Stil in development process</h3>**


## Bugs
-----------
* For consistency_loss, I will use sobel filter but opencv's sobel filter function is not support batched input.
* The "scipy import ndimage" is using sobel filter with numpy array's it can maybe support batched input and try
* There is another bug in this situation, because in dataset.py you are converting the masked image to 2 channel image because of the num of classification number. The opencv's Sobel filter is not support dimensions which is under 3 channels. 
-----------