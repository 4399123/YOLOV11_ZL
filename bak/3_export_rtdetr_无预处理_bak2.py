import torch
import torch.nn as nn
from ultralytics import RTDETR
import onnx
from onnxsim import simplify
import argparse
import os


# 定义终极封装类
class End2EndModel(nn.Module):
    def __init__(self, model, topk=150):
        super().__init__()
        self.model = model
        # RT-DETR 架构特性：查询数固定为 300
        # 我们在初始化时就确定 k 的值，避免在 forward 里动态计算
        # 这样 k 就是一个纯 Python 整数，不会生成 ONNX 的 Min/If 节点
        self.max_queries = 300
        self.k = min(topk, self.max_queries)

    def forward(self, x):
        # ---------------- 1. 预处理 ----------------
        x = x.float() / 255.0
        x = x[:, [2, 1, 0], :, :]  # BGR -> RGB

        # ---------------- 2. 模型推理 ----------------
        y = self.model(x)

        if isinstance(y, (list, tuple)):
            y = y[0]
        # y shape: [Batch, 300, 4 + NumClasses]

        # ---------------- 3. 数据拆分 ----------------
        boxes = y[:, :, :4]
        class_scores = y[:, :, 4:]

        # ---------------- 4. 计算最大得分 ----------------
        # max_scores: [Batch, 300]
        # class_ids:  [Batch, 300]
        max_scores, class_ids = torch.max(class_scores, dim=2)

        # ---------------- 5. Top-K 排序 (关键修改) ----------------
        # 直接使用 self.k (它是一个 int 常量)，不要使用 y.shape[1]
        # 这样 TensorRT 就知道输出形状是固定的 [Batch, k, 6]
        topk_scores, topk_indices = torch.topk(max_scores, self.k, dim=1, sorted=True)

        # ---------------- 6. 根据索引提取数据 (Gather) ----------------
        indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, 4)
        topk_boxes = torch.gather(boxes, 1, indices_expanded)

        topk_class_ids = torch.gather(class_ids, 1, topk_indices)

        # ---------------- 7. 拼接最终输出 ----------------
        topk_scores = topk_scores.unsqueeze(-1)
        topk_class_ids = topk_class_ids.unsqueeze(-1).float()

        # [B, k, 6] -> [cx, cy, w, h, score, class_id]
        final_output = torch.cat([topk_boxes, topk_scores, topk_class_ids], dim=2)

        return final_output


def main(weight_path, onnx_best_path, onnx_sim_path, topk):
    print(f"Loading model from {weight_path}...")
    base_model = RTDETR(weight_path)

    print("Fusing model layers...")
    base_model.model.fuse()
    pytorch_model = base_model.model.eval().cpu()

    # 封装模型
    print(f"Wrapping model with Top-{topk} sorting...")
    wrapped_model = End2EndModel(pytorch_model, topk=topk)
    wrapped_model.eval()

    dummy_input = torch.zeros(2, 3, 640, 640, dtype=torch.uint8, device='cpu')

    print("Exporting to ONNX (Opset 16)...")
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        onnx_best_path,
        # --- 必须使用 16 以支持 grid_sampler ---
        opset_version=16,
        # --- 建议开启常量折叠，因为我们消除了动态 k ---
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f'Original ONNX saved to {onnx_best_path}')

    print("Simplifying ONNX...")
    model_onnx = onnx.load(onnx_best_path)

    # 尝试简化
    try:
        model_simp, check = simplify(model_onnx)
        if check:
            onnx.save(model_simp, onnx_sim_path)
            print(f'Simplified ONNX saved to {onnx_sim_path}')
        else:
            print("Check failed! Saving unverified model...")
            onnx.save(model_simp, onnx_sim_path)
    except Exception as e:
        print(f"Simplification failed: {e}")
        print("Saving original ONNX as the final result.")
        onnx.save(model_onnx, onnx_sim_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight-path', type=str, default='pt/baofeng/v1/rtdetr-l.pt')
    parser.add_argument('--outpath', type=str, default='./onnx/best.onnx')
    parser.add_argument('--ousmitpath', type=str, default='./onnx/best-smi.onnx')
    parser.add_argument('--topk', type=int, default=150)

    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.outpath), exist_ok=True)
    main(args.weight_path, args.outpath, args.ousmitpath, args.topk)