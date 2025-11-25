import sys
import os
import torch
import torch.nn as nn
from ultralytics import YOLO
import onnx
from onnxsim import simplify
import argparse

# 获取当前脚本的目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


class YOLOv10PostProcess(nn.Module):
    def __init__(self, model, topk=100, img_sz=640):
        super().__init__()
        self.model = model
        self.topk = int(topk)  # 确保是 int
        self.img_sz = float(img_sz)

    def forward(self, x):
        # -------------------------
        # 1. 预处理 (Pre-process)
        # -------------------------
        # x: uint8 BGR [B, 3, H, W] -> float RGB [B, 3, H, W]
        x = x.float() / 255.0
        x = x[:, [2, 1, 0], ...]

        # -------------------------
        # 2. 模型推理 (Inference)
        # -------------------------
        # YOLOv10 原生输出: [Batch, 300, 6] (x1, y1, x2, y2, conf, cls)
        preds = self.model(x)

        if isinstance(preds, (tuple, list)):
            preds = preds[0]

        # -------------------------
        # 3. 排序与筛选 (Sort & TopK)
        # -------------------------
        # 取出 conf (第4列)
        scores = preds[:, :, 4]

        # 【核心修复】使用 torch.topk 替代 torch.sort
        # 我们直接取前 topk 个，而不是给所有 300 个排序
        # 传入的 k=self.topk 是一个 python int 常量，这会让 ONNX 中的 K 变成 Initializer
        topk_scores, topk_indices = torch.topk(scores, self.topk, dim=1, largest=True, sorted=True)

        # 收集对应的数据 [B, K, 6]
        # 扩展 indices 维度以匹配 gather 需求: [B, K] -> [B, K, 6]
        indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, 6)

        # 这里的 preds 是 [B, 300, 6]，gather 会按照 indices 挑出我们需要的行
        selected_preds = torch.gather(preds, 1, indices_expanded)

        # -------------------------
        # 4. 坐标归一化 & 格式转换
        # -------------------------
        # 原始数据是绝对坐标 (像素值)
        x1_abs = selected_preds[:, :, 0]
        y1_abs = selected_preds[:, :, 1]
        x2_abs = selected_preds[:, :, 2]
        y2_abs = selected_preds[:, :, 3]
        conf = selected_preds[:, :, 4]
        cls = selected_preds[:, :, 5]

        # 归一化 (0.0 - 1.0)
        x1 = x1_abs / self.img_sz
        y1 = y1_abs / self.img_sz
        x2 = x2_abs / self.img_sz
        y2 = y2_abs / self.img_sz

        # 转换为 cx, cy, w, h
        w_norm = x2 - x1
        h_norm = y2 - y1
        cx_norm = (x1 + x2) / 2.0
        cy_norm = (y1 + y2) / 2.0

        # 堆叠输出 [B, K, 6]
        # Format: [cx_norm, cy_norm, w_norm, h_norm, score, class_id]
        final_output = torch.stack([cx_norm, cy_norm, w_norm, h_norm, conf, cls], dim=2)

        return final_output


def main(weight_path, onnx_path, topk, img_sz):
    print(f"Loading YOLOv10 model from {weight_path}...")

    yolo = YOLO(weight_path)
    model = yolo.model

    print("Fusing model layers...")
    model.fuse()
    model.eval()

    print(f"Wrapping model (Top-{topk}, Normalized Coords)...")
    wrapped = YOLOv10PostProcess(model, topk=topk, img_sz=img_sz)

    print("Exporting to ONNX...")
    # 输入尺寸必须和 img_sz 一致
    dummy = torch.zeros(1, 3, img_sz, img_sz, dtype=torch.uint8)

    torch.onnx.export(
        wrapped,
        dummy,
        onnx_path,
        opset_version=11,  # 保持 11
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f'Raw ONNX exported to {onnx_path}')

    print("Simplifying ONNX...")
    model_onnx = onnx.load(onnx_path)
    model_simp, check = simplify(model_onnx)
    assert check, "Simplified ONNX model could not be validated"

    onnx.save(model_simp, onnx_path)
    print(f'Success! Optimized ONNX saved to {onnx_path}')
    print(f'Output Shape: [Batch, {topk}, 6]')


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--weight-path', dest='weight_pth', type=str,
                       default='../pt/baofeng/v1/yolov10-s.pt')
    parse.add_argument('--outpath', dest='out_pth', type=str,
                       default='../onnx/best-smi.onnx')
    parse.add_argument('--topk', dest='topk', type=int, default=150,
                       help='Output top K boxes')
    parse.add_argument('--imgsz', dest='imgsz', type=int, default=640,
                       help='Inference image size')

    args = parse.parse_args()

    main(args.weight_pth, args.out_pth, args.topk, args.imgsz)