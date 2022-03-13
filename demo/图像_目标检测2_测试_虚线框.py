import cv2
import torch
import  numpy as np
import matplotlib.pyplot as plt
from ssd import build_ssd
from data import VOC_CLASSES as labels
from matplotlib import patches
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
use_gpu=torch.cuda.is_available()

if use_gpu:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

frame=cv2.imread('1.jpg')
# f=cv2.imread('2.png')
cv2.waitKey(0)

frame_o=frame.copy()
frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
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
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth',map_location=torch.device('cpu')))



with torch.no_grad():
    net.eval()
    y = net(xx)


    fig=plt.figure()
    colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
    # plt.subplot(133)
    # plt.subplot(111)
      # plot the image for matplotlib
    currentAxis = plt.gca()

    detections = y.data
    # scale each detection back up to the image
    scale = torch.Tensor(frame_o.shape[1::-1]).repeat(2)
    for i in range(detections.size(1)):
        j = 0                                                          # '-'or'solid'      实线
        while detections[0, i, j, 0] >= 0.6:                           # '--'or'dashed'    虚线
            score = detections[0, i, j, 0]                             # '-.'or'dashdot'   点划线
            label_name = labels[i - 1]                                 # ':'or'dotted'     点
            display_txt = '%s: %.2f' % (label_name, score)
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
            coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1
            color = colors[i]
            currentAxis.add_patch(patches.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2,linestyle="--",alpha=0.8,facecolor=color))
            currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor': color, 'alpha': 0.5})
            j += 1
    plt.imshow(frame)
    ###
    # fig.canvas.draw()
    # X = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    # X = X.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    # X = X[..., ::-1]
    # cv2.imshow('frame',X)
    # cv2.waitKey(0)
    ###
# plt.subplot(131)
# plt.imshow(frame_o)
# plt.subplot(132)
# plt.imshow(frame_o2)
plt.show()
