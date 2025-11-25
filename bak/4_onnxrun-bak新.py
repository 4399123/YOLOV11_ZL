#encoding=gbk
import os.path

import onnx
import onnxruntime as ort
import numpy as np
import cv2
from tqdm import  tqdm
from imutils import paths
from letterBOX import letterbox
#路径配置
onnx_path=r'./pt/bftuan/v2/bf_yolo11_n.onnx'
imgspath=r'C:\G\Baofeng\proj\3_det_seg\all_imgs\imgs1'
w,h=640,640

score_threshold=0.5
nms_threshold=0.3

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


label={0:'TA',
       1:'LV',
}

# label={0:'BOX'}


imgpaths=list(paths.list_images(imgspath))

#onnx模型载入
model = onnx.load(onnx_path)
onnx.checker.check_model(model)
session = ort.InferenceSession(onnx_path,providers=['CPUExecutionProvider'])


for pic_path in tqdm(imgpaths):
    basename=os.path.basename(pic_path)
    img=cv2.imread(pic_path)
    imgbak=img.copy()
    img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img=cv2.resize(img,(w,h))
    img,r,dw,dh=letterbox(img,h,w)
    img=img.astype(np.float32)
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
    indices = cv2.dnn.NMSBoxes(bboxes=boxes, scores=scores, score_threshold=score_threshold, nms_threshold=nms_threshold)
    for i in indices:
        box = boxes[i]
        id=class_ids[i]
        score=scores[i]
        # x1=int(box[0]*w_ratio)
        # y1=int(box[1]*h_ratio)
        # x2=int((box[0]+box[2])*w_ratio)
        # y2=int((box[1]+box[3])*h_ratio)
        x1=int((box[0]-dw)/r)
        y1=int((box[1]-dh)/r)
        x2=int(((box[0]+box[2])-dw)/r)
        y2=int(((box[1]+box[3])-dh)/r)
        color=palette[id]
        cv2.rectangle(imgbak, (x1, y1), (x2, y2), color, 2)
        cv2.putText(imgbak, '{}:{:.6f}'.format(label[int(id)], float(score)), (x1, y1 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.2, palette[int(id)], 1)

    cv2.imwrite('./results/{}_res.jpg'.format(basename),imgbak)









