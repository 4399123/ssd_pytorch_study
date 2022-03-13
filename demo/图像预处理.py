import cv2
import torch
from PIL import Image
import  numpy as np
import matplotlib.pyplot as plt
############################################
# x=cv2.imread('1.jpg')
# x=cv2.resize(x,(300,300)).astype(np.float32)
# x -= (104.0, 117.0, 123.0)
# x = x[:, :, ::-1]
# plt.imshow(x)
# plt.show()

##########################
# x=cv2.cvtColor(cv2.imread('1.jpg'),cv2.COLOR_BGR2RGB)
# x=cv2.resize(x,(300,300)).astype(np.float32)
# x-=(104.0, 117.0, 123.0)
# plt.imshow(x)
# plt.show()

##################
x=cv2.cvtColor(cv2.imread('1.jpg'),cv2.COLOR_BGR2RGB)
x1=cv2.resize(x,(300,300))
x=x1.astype(np.float32)
#
plt.subplot(1,2,1)
plt.title('原图')
plt.axis('off')
plt.imshow(x1)
#
x -= (104.0, 117.0, 123.0)
# x/=255.0
plt.subplot(1,2,2)
plt.title('处理后的图')
plt.axis('off')
plt.imshow(x)
plt.show()



