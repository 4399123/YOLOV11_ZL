import sys
import os

# 获取当前脚本的目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取上一级目录（project/）的路径
parent_dir = os.path.dirname(current_dir)
# 将上一级目录添加到 Python 搜索路径
sys.path.append(parent_dir)

import warnings
import os
warnings.filterwarnings('ignore')
from ultralytics import YOLO,RTDETR
if __name__ == '__main__':
    # model = YOLO('./ultralytics/cfg/models/11/yolo11.yaml')   # 修改yaml
    # model.load('./weights/yolo11s.pt')  #加载预训练权重
    model = YOLO('./weights/yolov10s.pt')
    # model =RTDETR('./weights/rtdetr-l.pt')
    model.train(
        data=r'./data/bftuan.yaml',
        epochs=300,  # (int) 训练的周期数
        batch=16,  # (int) 每批次的图像数量（-1 为自动批处理）
        imgsz=[640,640],  # (int) 输入图像的大小，整数或w，h
        save_period=-1,  # (int) 每x周期保存检查点（如果小于1则禁用）
        cache=True,  # (bool) True/ram、磁盘或False。使用缓存加载数据
        device='0,1',  # (int | str | list, optional) 运行的设备，例如 cuda device=0 或 device=0,1,2,3 或 device=cpu
        project='runs/train',  # (str, optional) 项目名称
        name='exp',  # (str, optional) 实验名称，结果保存在'project/name'目录下
        exist_ok=False,  # (bool) 是否覆盖现有实验
        pretrained=True,  # (bool | str) 是否使用预训练模型（bool），或从中加载权重的模型（str）
        optimizer='auto',  # (str) 要使用的优化器，选择=[SGD，Adam，Adamax，AdamW，NAdam，RAdam，RMSProp，auto]
        verbose=True,  # (bool) 是否打印详细输出
        seed=42,  # (int) 用于可重复性的随机种子
        cos_lr=True,  # (bool) 使用余弦学习率调度器
        close_mosaic=10,  # (int) 在最后几个周期禁用马赛克增强
        resume=False,  # (bool) 从上一个检查点恢复训练
        amp=True,  # (bool) 自动混合精度（AMP）训练，选择=[True, False]，True运行AMP检查
        # 超参数 ----------------------------------------------------------------------------------------------
        # lr0=0.0005,  # (float) 初始学习率（例如，SGD=1E-2，Adam=1E-3）
        # lrf=0.0001,  # (float) 最终学习率（lr0 * lrf）
        # momentum=0.937,  # (float) SGD动量/Adam beta1
        # weight_decay=0.0005,  # (float) 优化器权重衰减 5e-4
        # warmup_epochs=3.0,  # (float) 预热周期（分数可用）
        # warmup_momentum=0.8,  # (float) 预热初始动量
        # warmup_bias_lr=0.1,  # (float) 预热初始偏置学习率
        workers=8,
        #数据增强
        hsv_h=0.015,  # (float) image HSV-Hue augmentation (fraction)
        hsv_s=0.7,  # (float) image HSV-Saturation augmentation (fraction)
        hsv_v=0.4,  # (float) image HSV-Value augmentation (fraction)
        degrees=5,  # (float) image rotation (+/- deg)
        translate=0.1,  # (float) image translation (+/- fraction)
        scale=0.2 , # (float) image scale (+/- gain)
        shear=0.1,  # (float) image shear (+/- deg)
        perspective=0.0,  # (float) image perspective (+/- fraction), range 0-0.001
        flipud=0.05,  # (float) image flip up-down (probability)
        fliplr=0.3,  # (float) image flip left-right (probability)
        bgr=0.1,  # (float) image channel BGR (probability)
        mosaic=0.5,  # (float) image mosaic (probability)
        mixup=0.1,  # (float) image mixup (probability)
        copy_paste=0.0,  # (float) segment copy-paste (probability)
    )



