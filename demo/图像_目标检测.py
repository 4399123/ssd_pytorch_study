import cv2
import torch
import  numpy as np
import matplotlib.pyplot as plt
from ssd import build_ssd
from data import VOC_CLASSES as labelmap

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
use_gpu=torch.cuda.is_available()

if use_gpu:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

frame=cv2.imread('1.jpg')
frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
frame_o=frame.copy()
height, width=frame.shape[:2]
x=cv2.resize(frame,(300,300))
x=x.astype(np.float32)
x -= (104.0, 117.0, 123.0)
frame_o2=x.copy()
x=torch.from_numpy(x).permute(2,0,1)

if use_gpu:
    xx=torch.autograd.Variable(torch.unsqueeze(x,0)).cuda()
else:
    xx = torch.autograd.Variable(torch.unsqueeze(x, 0))



net = build_ssd('test', 300, 21) # initialize SSD
# net = net.to(device="cuda")

net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth'))

with torch.no_grad():
    net.eval()
    y = net(xx)
    detections = y.data
    # scale each detection back up to the image
    scale = torch.Tensor([width, height, width, height])
    for i in range(detections.size(1)):

        j = 0
        while detections[0, i, j, 0] >= 0.6:
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
            cv2.rectangle(frame,
                          (int(pt[0]), int(pt[1])),
                          (int(pt[2]), int(pt[3])),
                          COLORS[i % 3], 2)
            cv2.putText(frame, labelmap[i - 1], (int(pt[0]), int(pt[1])-5),
                        0, 1, (255, 255, 255), 2, cv2.LINE_AA)
            j += 1


plt.subplot(131)
plt.imshow(frame_o)
plt.subplot(132)
plt.imshow(frame_o2)
plt.subplot(133)
plt.imshow(frame)
plt.show()
