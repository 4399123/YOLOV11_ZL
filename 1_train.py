import warnings
import os
warnings.filterwarnings('ignore')
from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO('./ultralytics/cfg/models/v8/yolov8n.yaml')   # 修改yaml
    model.load('./weights/yolov8n.pt')  #加载预训练权重
    model.train(
        data=r'./data/catdog.yaml',
        epochs=300,  # (int) 训练的周期数
        batch=32,  # (int) 每批次的图像数量（-1 为自动批处理）
        imgsz=[640,640],  # (int) 输入图像的大小，整数或w，h
        save_period=-1,  # (int) 每x周期保存检查点（如果小于1则禁用）
        cache=True,  # (bool) True/ram、磁盘或False。使用缓存加载数据
        device='0,1',  # (int | str | list, optional) 运行的设备，例如 cuda device=0 或 device=0,1,2,3 或 device=cpu
        project='runs/train',  # (str, optional) 项目名称
        name='exp',  # (str, optional) 实验名称，结果保存在'project/name'目录下
        exist_ok=False,  # (bool) 是否覆盖现有实验
        pretrained=True,  # (bool | str) 是否使用预训练模型（bool），或从中加载权重的模型（str）
        optimizer='SGD',  # (str) 要使用的优化器，选择=[SGD，Adam，Adamax，AdamW，NAdam，RAdam，RMSProp，auto]
        verbose=True,  # (bool) 是否打印详细输出
        seed=42,  # (int) 用于可重复性的随机种子
        cos_lr=True,  # (bool) 使用余弦学习率调度器
        close_mosaic=10,  # (int) 在最后几个周期禁用马赛克增强
        resume=False,  # (bool) 从上一个检查点恢复训练
        amp=True,  # (bool) 自动混合精度（AMP）训练，选择=[True, False]，True运行AMP检查
        # 超参数 ----------------------------------------------------------------------------------------------
        lr0=0.0002,  # (float) 初始学习率（例如，SGD=1E-2，Adam=1E-3）
        lrf=0.01,  # (float) 最终学习率（lr0 * lrf）
        momentum=0.937,  # (float) SGD动量/Adam beta1
        weight_decay=0.0005,  # (float) 优化器权重衰减 5e-4
        warmup_epochs=3.0,  # (float) 预热周期（分数可用）
        warmup_momentum=0.8,  # (float) 预热初始动量
        warmup_bias_lr=0.1,  # (float) 预热初始偏置学习率
        workers=0
    )
