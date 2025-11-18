#encoding=gbk
import os.path

import onnx
import onnxruntime as ort
import numpy as np
import cv2
from tqdm import  tqdm
from imutils import paths
#路径配置
onnx_path=r'./pt/catdog/11_n.onnx'
imgspath=r'C:\D\github_zl\CatDogDetDataSetV2/'
w,h=640,640

score_threshold=0.25
nms_threshold=0.45

if not os.path.exists('./results'):
    os.makedirs('./results')

palette={0:(0,255,0),
    1:(0,0,255),
    2:(255,0,0),
    3:(255,255,0),
    4:(255,0,255),
    5:(171,130,255),
    6:(155,211,255),
    7:(0,255,255)}


label={0:'cat',
       1:'dog',
       2:'eagle',
       3:'elephant'}

# label={0:'BOX'}


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
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
        if maxScore >= score_threshold:
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
    result_boxes = cv2.dnn.NMSBoxes(bboxes=boxes, scores=scores, score_threshold=score_threshold, nms_threshold=nms_threshold)
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
        cv2.rectangle(imgbak, (x1, y1), (x2, y2), color, 2)
        cv2.putText(imgbak, '{}:{:.2f}'.format(label[int(id)], float(score)), (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, palette[int(id)], 1)

    cv2.imwrite('./results/{}_res.jpg'.format(basename),imgbak)









