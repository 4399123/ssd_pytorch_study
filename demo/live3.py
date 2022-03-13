from __future__ import print_function
import torch
from torch.autograd import Variable
import cv2
import time
import argparse
from numpy import random

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

colors = [[random.randint(0, 256) for _ in range(3)] for _ in names]
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
        ##############
        # imshow()显示图像时对double型是认为在0~1范围内。即大于1时都是显示白色，而imshow显示uint8型时
        # 是0~255范围内。而经过运算的范围在0~255之间的double型数据就被不正常得显示为白色图像了
        # 解决方法：imshow(I/256); -----------将图像矩阵转化到0-1之间
        f=transform(frame)[0]
        cv2.imshow('f', f)
        ##############
        x = torch.from_numpy(f).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0)).cuda()
        y = net(x)# forward pass
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([width, height, width, height])
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.5:
                score=detections[0, i, j, 0]
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                cv2.rectangle(frame,
                              (int(pt[0]), int(pt[1])),
                              (int(pt[2]), int(pt[3])),
                              colors[i-1], 2)
                #####
                cv2.rectangle(frame,
                              (int(pt[0]), int(pt[1])),
                              (int(pt[2]), int(pt[1])-15),
                              colors[i-1], thickness=-1)
                #####
                cv2.putText(frame,'{}:{:.2}'.format(labelmap[i - 1],score) , (int(pt[0]), int(pt[1])),#因为背景也有标签，所以要减去1
                            FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                j += 1
        f_end=time.time()
        fps='{:.1f}fps'.format(1/(f_end-f_start))
        cv2.putText(frame, fps, (20,50),FONT, 1, (0, 0, 0), 2, cv2.LINE_AA) ###cv2.LINE_AA 抗锯齿
        return frame

    # start video stream thread, allow buffer to fill
    print("[INFO] starting threaded video stream...")
    stream=cv2.VideoCapture(0)
    assert stream.isOpened(), f'摄像头打不开'
    stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    time.sleep(1.0)

    while True:
        # grab next frame
        re,frame = stream.read()
        assert  re,'无法获取视频帧'
        frame = predict(frame)
        cv2.imshow('frame',frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

if __name__ == '__main__':
    from data import BaseTransform, VOC_CLASSES as labelmap
    from ssd import build_ssd

    net = build_ssd('test', 300, 21) # initialize SSD
    net.load_state_dict(torch.load(args.weights))
    transform = BaseTransform(net.size, (104, 117, 123))

    with torch.no_grad():
        cv2_demo(net.eval(), transform)
    # cleanup
    cv2.destroyAllWindows()

