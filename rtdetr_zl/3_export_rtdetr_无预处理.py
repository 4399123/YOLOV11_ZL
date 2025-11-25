import sys
import os

# 获取当前脚本的目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取上一级目录（project/）的路径
parent_dir = os.path.dirname(current_dir)
# 将上一级目录添加到 Python 搜索路径
sys.path.append(parent_dir)

import torch
import torch.nn as nn
from ultralytics import RTDETR
import onnx
import argparse
import os


# ----------------------------------------------------------------
# 1. 模型定义 (支持配置类别数)
# ----------------------------------------------------------------
class End2EndModel(nn.Module):
    def __init__(self, model, topk=150, num_classes=2):
        super().__init__()
        self.model = model
        self.max_queries = 300
        self.k = int(min(topk, self.max_queries))

        # 【关键修改】动态计算通道数
        # 4个坐标 (cx, cy, w, h) + N个类别分数
        self.num_channels = 4 + num_classes
        print(f"[Info] Configured for {num_classes} classes. Total channels per box: {self.num_channels}")

    def forward(self, x):
        # 1. 预处理
        x = x.float() / 255.0
        x = x[:, [2, 1, 0], :, :]

        # 2. 推理
        y = self.model(x)
        if isinstance(y, (list, tuple)):
            y = y[0]

        # ---------------------------------------------------------
        # 【核心修复】强制维度重整
        # 使用动态计算的 self.num_channels 进行 Reshape
        # ---------------------------------------------------------
        y = y.reshape(-1, 300, self.num_channels)

        # 3. 后处理
        boxes = y[:, :, :4]
        class_scores = y[:, :, 4:]

        max_scores, class_ids = torch.max(class_scores, dim=2)

        # TopK
        topk_scores, topk_indices = torch.topk(max_scores, self.k, dim=1, sorted=True)

        indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, 4)
        topk_boxes = torch.gather(boxes, 1, indices_expanded)
        topk_class_ids = torch.gather(class_ids, 1, topk_indices)

        topk_scores = topk_scores.unsqueeze(-1)
        topk_class_ids = topk_class_ids.unsqueeze(-1).float()

        final_output = torch.cat([topk_boxes, topk_scores, topk_class_ids], dim=2)
        return final_output


# ----------------------------------------------------------------
# 2. ONNX 手术 (必不可少，保持不变)
# ----------------------------------------------------------------
def fix_onnx_if_nodes(onnx_path, output_path):
    print(f"\n[Surgery] Opening {onnx_path} for surgery...")
    model = onnx.load(onnx_path)
    graph = model.graph

    nodes_to_remove = []

    for node in graph.node:
        if node.op_type == "If":
            then_branch = next((attr.g for attr in node.attribute if attr.name == "then_branch"), None)
            else_branch = next((attr.g for attr in node.attribute if attr.name == "else_branch"), None)

            if then_branch and else_branch:
                then_ops = [n.op_type for n in then_branch.node]
                else_ops = [n.op_type for n in else_branch.node]

                if "Squeeze" in then_ops and "Identity" in else_ops:
                    print(f"[Surgery] Found culprit If node: {node.name}")

                    identity_node = [n for n in else_branch.node if n.op_type == "Identity"][0]
                    bypass_source_name = identity_node.input[0]
                    if_output_name = node.output[0]

                    print(f"   -> Rewiring: '{if_output_name}' -> '{bypass_source_name}'")

                    for other_node in graph.node:
                        for idx, inp in enumerate(other_node.input):
                            if inp == if_output_name:
                                other_node.input[idx] = bypass_source_name

                    nodes_to_remove.append(node)

    for node in nodes_to_remove:
        graph.node.remove(node)

    print(f"[Surgery] Removed {len(nodes_to_remove)} nodes.")
    print(f"[Surgery] Saving fixed model to {output_path}...")
    onnx.save(model, output_path)


def main(weight_path, onnx_path, topk, num_classes):
    print(f"Loading model from {weight_path}...")
    base_model = RTDETR(weight_path)
    base_model.model.fuse()
    pytorch_model = base_model.model.eval().cpu()

    # 传入 num_classes
    print(f"Wrapping model with Top-{topk} sorting, Classes={num_classes}...")
    wrapped_model = End2EndModel(pytorch_model, topk=topk, num_classes=num_classes)

    dummy_input = torch.zeros(1, 3, 640, 640, dtype=torch.uint8, device='cpu')
    temp_onnx = onnx_path.replace(".onnx", "_temp.onnx")

    print("Exporting ONNX (Opset 16, Folding=True)...")
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        temp_onnx,
        opset_version=16,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
    )

    # 执行手术
    fix_onnx_if_nodes(temp_onnx, onnx_path)

    if os.path.exists(temp_onnx):
        os.remove(temp_onnx)

    print(f"\nSUCCESS! Use '{onnx_path}' for TensorRT.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight-path', type=str, default='../pt/baofeng/v1/rtdetr-l.pt')
    parser.add_argument('--outpath', type=str, default='../onnx/best_final.onnx')
    parser.add_argument('--topk', type=int, default=150, help='Number of output boxes')
    # 新增参数
    parser.add_argument('--num-classes', type=int, default=2, help='Number of classes in your dataset (default: 2)')

    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.outpath), exist_ok=True)

    main(args.weight_path, args.outpath, args.topk, args.num_classes)