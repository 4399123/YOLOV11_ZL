#encoding=gbk
import os.path

import onnx
import onnxruntime as ort
import numpy as np
import cv2
from tqdm import  tqdm
from imutils import paths
#路径配置
onnx_path=r'./pt/8_n.onnx'
imgspath=r'./images/'
w,h=640,640

if not os.path.exists('./results'):
    os.makedirs('./results')

palette={0:(0,255,0),
    1:(0,0,255),
    2:(255,0,0),}

label={0:'BOX'}


imgpaths=list(paths.list_images(imgspath))

#onnx模型载入
model = onnx.load(onnx_path)
onnx.checker.check_model(model)
session = ort.InferenceSession(onnx_path,providers=['CPUExecutionProvider'])


for pic_path in tqdm(imgpaths):
    basename=os.path.basename(pic_path)
    img=cv2.imread(pic_path)
    H,W=img.shape[0],img.shape[1]
    h_ratio=H/h
    w_ratio=W/w
    imgbak=img.copy()
    img=img[:,:,::-1]
    img=cv2.resize(img,(w,h)).astype(np.float32)
    img = img / 255.0
    img=np.array([np.transpose(img,(2,0,1))])


    #模型推理
    outputs = session.run(None,input_feed = { 'images' : img })

    outputs = np.array([cv2.transpose(outputs[0][0])])
    rows = outputs.shape[1]

    boxes = []
    scores = []
    class_ids = []

    # Iterate through output to collect bounding boxes, confidence scores, and class IDs
    for i in range(rows):
        classes_scores = outputs[0][i][4:]
        (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
        if maxScore >= 0.25:
            box = [
                float(outputs[0][i][0] - (0.5 * outputs[0][i][2])),  # x center - width/2 = left x
                float(outputs[0][i][1] - (0.5 * outputs[0][i][3])),  # y center - height/2 = top y
                float(outputs[0][i][2]),  # width
                float(outputs[0][i][3]),  # height
            ]
            boxes.append(box)
            scores.append(maxScore)
            class_ids.append(maxClassIndex)

    # Apply NMS (Non-maximum suppression)
    result_boxes = cv2.dnn.NMSBoxes(bboxes=boxes, scores=scores, score_threshold=0.25, nms_threshold=0.45)
    for i in range(len(result_boxes)):
        index = result_boxes[i][0]
        box = boxes[index]
        id=class_ids[index]
        score=scores[index]
        x1=int(box[0]*w_ratio)
        y1=int(box[1]*h_ratio)
        x2=int((box[0]+box[2])*w_ratio)
        y2=int((box[1]+box[3])*h_ratio)
        color=palette[id]
        label=f"{label[id]} {round(score*100,2)}%"
        cv2.rectangle(imgbak, (x1, y1), (x2, y2), color, 2)
    cv2.imwrite('./results/{}_res.jpg'.format(basename),imgbak)









