from __future__ import print_function
import torch
from torch.autograd import Variable
import cv2
import time
from imutils.video import FPS, WebcamVideoStream
import argparse
from numpy import random
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--weights', default='ssd300_mAP_77.43_v2.pth',
                    type=str, help='Trained state_dict file path')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda in live demo')
args = parser.parse_args()


names=['aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor']

# COLORS = [[255, 0, 0], (0, 255, 0), (0, 0, 255)]
# colors = [[random.randint(0, 256) for _ in range(3)] for _ in names]
colors=plt.cm.viridis(np.linspace(0,1,20)).tolist()
# print(colors*255)
###
# colorss=[]
# for _ in names:
#     c=[]
#     for _ in range(3):
#         c.append(random.randint(0,255))
#     colorss.append(c)
# print(colorss)

#####



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
        # scale each detection back up to the image
        scale = torch.Tensor([width, height, width, height])
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.1:
                score=detections[0, i, j, 0]
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                color=[int((colors[i][kk]) * 255) for kk in range(3)]
                cv2.rectangle(frame,
                              (int(pt[0]), int(pt[1])),
                              (int(pt[2]), int(pt[3])),
                              color, 2)
                #####
                # print(colors[i-1])
                # print([int((colors[i - 1][kk])*255)for kk in range(3)])

                cv2.rectangle(frame,
                              (int(pt[0]), int(pt[1])),
                              (int(pt[2]), int(pt[1])-15),
                              color, thickness=-1)
                #####
                cv2.putText(frame,'{}:{:.2}'.format(labelmap[i - 1],score) , (int(pt[0]), int(pt[1])),
                            FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                j += 1
        f_end=time.time()
        fps='{}fps'.format(int(1/(f_end-f_start)))
        cv2.putText(frame, fps, (20,30),
                    FONT, 1, (255, 255, 255), 2, cv2.LINE_AA)
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
        _,frame = stream.read()
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
    transform = BaseTransform(net.size, (104, 117, 123))

    fps = FPS().start()
    with torch.no_grad():
        cv2_demo(net.eval(), transform)
    # stop the timer and display FPS information
    fps.stop()

    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # cleanup
    cv2.destroyAllWindows()

