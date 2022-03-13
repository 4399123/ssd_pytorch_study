from __future__ import print_function
import torch
from torch.autograd import Variable
import cv2
import time
from imutils.video import FPS, WebcamVideoStream
import argparse
from numpy import random
import cupy as cp
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--weights', default='ssd300_mAP_77.43_v2.pth',
                    type=str, help='Trained state_dict file path')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda in live demo')
args = parser.parse_args()


names=['aeroplane飞机', 'bicycle自行车', 'bird鸟类', 'boat船',
    'bottle瓶子', 'bus公交车', 'car小汽车', 'cat猫', 'chair椅子',
    'cow牛', 'diningtable餐桌', 'dog狗', 'horse马',
    'motorbike摩托车', 'person人类', 'pottedplant盘子',
    'sheep羊', 'sofa沙发', 'train火车', 'tvmonitor显示器']

# colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]


colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
# print(colors)
FONT = cv2.FONT_HERSHEY_SIMPLEX
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

def cv2_demo(net, transform):
    def predict(frame):
        f_start=time.time()
        height, width = frame.shape[:2]
        x = torch.from_numpy(transform(frame)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0)).cuda()
        y = net(x)# forward pass
        detections = y.data
        fig=plt.figure(figsize=(17,10))
        currentAxis = plt.gca()
        # scale each detection back up to the image
        scale = torch.Tensor([width, height, width, height])
        for i in range(detections.size(1)):
            j = 0                                                          # '-'or'solid'      实线
            while detections[0, i, j, 0] >= 0.5:                           # '--'or'dashed'    虚线
                score = detections[0, i, j, 0]                             # '-.'or'dashdot'   点划线
                label_name = names[i - 1]                                  # ':'or'dotted'     点
                display_txt = '%s: %.2f' % (label_name, score)
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1
                color = colors[i-1]
                currentAxis.add_patch(
                    patches.Rectangle(*coords, fill=True, edgecolor=color, linewidth=2, linestyle="--", alpha=0.2,
                                      facecolor=color))
                currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor': color, 'alpha': 0.5})
                j += 1
        # currentAxis.text(20, 30, '{}fps'.format(int(1/(eend-f_start))),fontdict={'size': 20, 'color':  'b'})
        # currentAxis.text(20, 30, '{}fps'.format(int(1 / (eend - f_start))), fontsize=20)
        plt.imshow(frame)
        plt.axis('off')
        plt.close()
        fig.canvas.draw()
        buf=fig.canvas.tostring_rgb()  #将RGB缓冲区存储到字符串中
        frame = np.fromstring(buf, dtype=np.uint8)
        # f_width,f_height=fig.canvas.get_width_height()         # f_width=col  f_height=row
        frame= frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        # frame = frame.reshape(f_height,f_width,3)           # reshape=(row,col)
        f_end = time.time()
        fps='{}fps'.format(int(1/(f_end-f_start)))
        cv2.putText(frame, fps, (20,30),
                    FONT, 1, (0, 0, 0), 2, cv2.LINE_AA)
        return frame

    # start video stream thread, allow buffer to fill
    print("[INFO] starting threaded video stream...")
    # stream = WebcamVideoStream(src=0).start()  # default camera
    stream=cv2.VideoCapture(0)
    assert stream.isOpened(), f'摄像头打不开'
    stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    time.sleep(1.0)
    # start fps timer
    # loop over frames from the video file stream
    while True:
        # grab next frame
        re,frame = stream.read()
        key = cv2.waitKey(1) & 0xFF

        # update FPS counter

        fps.update()

        frame = predict(frame)

        # keybindings for display
        if key == ord('p'):  # pause
            while True:
                key2 = cv2.waitKey(1) or 0xff
                cv2.imshow('frame', frame)
                if key2 == ord('p'):  # resume
                    break
        cv2.imshow('frame', frame)
        if key == 27:  # exit
            break


if __name__ == '__main__':
    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

    from data import BaseTransform, VOC_CLASSES as labelmap
    from ssd import build_ssd

    net = build_ssd('test', 300, 21) # initialize SSD
    # net = net.to(device="cuda")
    net.load_state_dict(torch.load(args.weights))
    transform = BaseTransform(net.size, (104/256.0, 117/256.0, 123/256.0))

    fps = FPS().start()
    with torch.no_grad():
        cv2_demo(net.eval(), transform)
    # stop the timer and display FPS information
    fps.stop()

    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # cleanup
    cv2.destroyAllWindows()

