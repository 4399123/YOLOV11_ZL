import torch
import torch.nn as nn
from ultralytics import RTDETR
import onnx
from onnxsim import simplify
import argparse
import os


# 定义预处理封装类
class PreProcessModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # x input: uint8 [B, C, H, W]  (注意：这里假设输入已经是NCHW格式)
        # 如果你的输入是 [B, H, W, C] (HWC)，需要先 permute

        # 1. 类型转换 uint8 -> float32 并归一化 0-1
        x = x.float() / 255.0

        # 2. BGR -> RGB (如果你确定输入是BGR，这一步是必须的)
        # x shape: [Batch, Channel, Height, Width]
        x = x[:, [2, 1, 0], :, :]

        # 3. 模型推理
        y = self.model(x)

        # 4. 清理输出 (关键步骤！)
        # Ultralytics 模型有时会返回 tuple/list，比如 (pred, hidden_states)
        # 我们只需要第一个元素作为最终输出
        if isinstance(y, (list, tuple)):
            return y[0]
        return y


def main(weight_path, onnx_best_path, onnx_sim_path):
    print(f"Loading model from {weight_path}...")

    # 1. 加载模型
    # 使用 ultralytics 接口加载，方便获取配置
    base_model = RTDETR(weight_path)

    # 2. 关键优化：Conv+BN 融合
    # 这会合并卷积和归一化层，提升推理速度
    print("Fusing model layers...")
    base_model.model.fuse()

    # 提取内部 PyTorch 模型并设为评估模式
    pytorch_model = base_model.model.eval().cpu()

    # 3. 封装预处理
    wrapped_model = PreProcessModel(pytorch_model)
    wrapped_model.eval()

    # 4. 准备 Dummy Input (CPU)
    # 模拟一张纯黑的图片，uint8 格式
    dummy_input = torch.zeros(1, 3, 640, 640, dtype=torch.uint8, device='cpu')

    print("Exporting to ONNX...")
    # 5. 导出 ONNX
    torch.onnx.export(
        wrapped_model,
        dummy_input,
        onnx_best_path,
        opset_version=16,  # RT-DETR 推荐 16 或 17
        do_constant_folding=True,  # 开启常量折叠
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f'Original ONNX saved to {onnx_best_path}')

    # 6. 使用 onnx-simplifier 进行简化
    print("Simplifying ONNX...")
    model_onnx = onnx.load(onnx_best_path)

    # simplify 已经包含了大部分必要的优化
    model_simp, check = simplify(model_onnx)

    if check:
        onnx.save(model_simp, onnx_sim_path)
        print(f'Simplified ONNX saved to {onnx_sim_path}')
        print("Done!")
    else:
        print("Check failed! Saving unverified simplified model...")
        onnx.save(model_simp, onnx_sim_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight-path', type=str, default='pt/baofeng/v1/rtdetr-l.pt', help='Path to .pt file')
    parser.add_argument('--outpath', type=str, default='./onnx/best.onnx', help='Intermediate ONNX path')
    parser.add_argument('--ousmitpath', type=str, default='./onnx/best-smi.onnx', help='Final simplified ONNX path')

    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.outpath), exist_ok=True)

    main(args.weight_path, args.outpath, args.ousmitpath)