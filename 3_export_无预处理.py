import torch
import torch.nn as nn
from ultralytics import YOLO
import onnx
from onnxsim import simplify
import onnxoptimizer
import argparse

class PreProcessModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # x: uint8 BGR  (OpenCV)
        x = x.float() / 255.0              # uint8 → float
        x = x[:,[2,1,0],...]              # BGR → RGB
        return self.model(x)




def main(modelpath,onnxbest,onnxsmi):

    # 1. 加载模型
    model = YOLO(modelpath).model

    # 2. 外面包一层预处理
    wrapped = PreProcessModel(model)

    # 3. 导出 ONNX
    dummy = torch.zeros(1, 3,640, 640, dtype=torch.uint8)  # 输入仍是 uint8
    torch.onnx.export(
        wrapped,
        dummy,
        onnxbest,
        opset_version=11,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print('step 1 ok')
    model = onnx.load(onnxbest)

    newmodel = onnxoptimizer.optimize(model)

    model_simp, check = simplify(newmodel)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, onnxsmi)
    print('step 2 ok')



if __name__ == '__main__':

    parse = argparse.ArgumentParser()
    parse.add_argument('--weight-path', dest='weight_pth', type=str,
                       default='pt/bftuan/v2/bf_yolo11_n.pt')          #训练好的模型路径
    parse.add_argument('--outpath', dest='out_pth', type=str,
                       default='./onnx/best.onnx')                     #中间产物onnx路径
    parse.add_argument('--ousmitpath', dest='outsmi_pth', type=str,
                       default='./onnx/best-smi.onnx')                 #最终产物onnx路径
    args = parse.parse_args()


    main(args.weight_pth,args.out_pth,args.outsmi_pth)
