# encoding=gbk
import os.path
import onnx
import onnxruntime as ort
import numpy as np
import cv2
from tqdm import tqdm
from imutils import paths
from letterBOX import letterbox

# --- 配置 ---
onnx_path = r'../onnx/best-smi.onnx'
imgspath = r'../images/baofeng'
w, h = 640, 640  # 模型输入尺寸
score_threshold = 0.45

if not os.path.exists('./results'):
    os.makedirs('./results')

palette = {
    0: (0, 255, 0), 1: (0, 0, 255), 2: (255, 0, 0), 3: (255, 255, 0),
    4: (255, 0, 255), 5: (171, 130, 255), 6: (155, 211, 255), 7: (0, 255, 255)
}

label = {
    0: 'TA',
    1: 'LV',
}

imgpaths = list(paths.list_images(imgspath))

# --- 加载模型 ---
print(f"Loading {onnx_path}...")
model = onnx.load(onnx_path)
onnx.checker.check_model(model)
session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

for pic_path in tqdm(imgpaths):
    basename = os.path.basename(pic_path)
    img = cv2.imread(pic_path)
    if img is None: continue

    # 1. 获取原图尺寸 (还原坐标时直接用这个)
    o_H, o_W = img.shape[:2]
    ratio_w=o_W/w
    ratio_h=o_H/h
    imgbak = img.copy()

    # --- 2. 预处理 (保持你的暴力拉伸写法) ---
    img = cv2.resize(img, (w, h))  # 强制拉伸到 640x640
    img, r, dw, dh = letterbox(img, h, w)  # r=1, dw=0

    # 构造输入 [1, 3, 640, 640] uint8
    img = np.array([np.transpose(img, (2, 0, 1))])
    img = np.ascontiguousarray(img, dtype=np.uint8)

    # --- 3. 推理 ---
    # outputs shape: [1, TopK, 6]
    outputs = session.run([output_name], {input_name: img})[0]
    output = outputs[0]  # [TopK, 6]

    # --- 4. 后处理 ---
    # row format: [cx, cy, w, h, score, class_id] (Normalized 0-1)
    for row in output:
        score = row[4]

        # 优化：因为是降序排列，低于阈值直接结束循环
        if score < score_threshold:
            break

        class_id = int(row[5])

        # 获取归一化坐标 (0.0 ~ 1.0)
        cx_norm, cy_norm, w_norm, h_norm = row[0], row[1], row[2], row[3]

        # --- 坐标还原 (极简版) ---
        # 直接乘以原图尺寸
        # cx_pixel = cx_norm * 原图宽
        cx = cx_norm * ratio_w*w
        cy = cy_norm * ratio_h*h
        bw = w_norm * ratio_w*w
        bh = h_norm * ratio_h*h

        # 转为左上/右下坐标
        x1 = int(cx - bw / 2)
        y1 = int(cy - bh / 2)
        x2 = int(cx + bw / 2)
        y2 = int(cy + bh / 2)

        # 边界保护
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(o_W, x2)
        y2 = min(o_H, y2)

        # --- 绘图 ---
        color = palette.get(class_id, (255, 0, 0))
        cv2.rectangle(imgbak, (x1, y1), (x2, y2), color, 3)

        label_text = label.get(class_id, str(class_id))
        cv2.putText(imgbak, '{}:{:.2f}'.format(label_text, float(score)), (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 2)
    basename=basename.replace('.png','.jpg')
    cv2.imwrite(f'./results/{basename}', imgbak)

print("Done.")