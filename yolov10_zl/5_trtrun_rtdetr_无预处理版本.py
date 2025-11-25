# encoding=gbk
import tensorrt as trt
import numpy as np
import os
import cv2
import pycuda.driver as cuda
import pycuda.autoinit
from imutils import paths
from tqdm import tqdm
from letterBOX import letterbox  # 确保目录下有这个文件


class TrtModel:
    def __init__(self, engine_path, max_batch_size=1):
        self.engine_path = engine_path
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(self.runtime, self.engine_path)
        self.max_batch_size = max_batch_size
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()

    @staticmethod
    def load_engine(trt_runtime, engine_path):
        trt.init_libnvinfer_plugins(None, "")
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        return trt_runtime.deserialize_cuda_engine(engine_data)

    def allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        for i in range(self.engine.num_bindings):
            binding = i
            dims = list(self.engine.get_binding_shape(binding))

            # 处理动态 Batch (-1 -> max_batch_size)
            if dims[0] == -1:
                dims[0] = self.max_batch_size

            size = trt.volume(dims)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))

            # 分配页锁定内存 (Host) 和 显存 (Device)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            bindings.append(int(device_mem))

            binding_info = {
                'host': host_mem,
                'device': device_mem,
                'shape': dims,
                'index': i,
                'dtype': dtype
            }

            if self.engine.binding_is_input(binding):
                inputs.append(binding_info)
            else:
                outputs.append(binding_info)

        return inputs, outputs, bindings, stream

    def __call__(self, x: np.ndarray):
        # x shape: (Batch, 3, 640, 640)
        batch_size = x.shape[0]

        # 1. 准备输入数据
        input_info = self.inputs[0]

        # 强制展平并拷贝到 Host 内存
        # 这一步非常关键：确保 numpy 数据类型与 Engine 输入类型一致 (通常是 uint8)
        x_flat = x.ravel().astype(input_info['dtype'])
        np.copyto(input_info['host'][:x_flat.size], x_flat)

        # 2. Host -> Device
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)

        # 3. 设置动态 Input Shape
        self.context.set_binding_shape(self.inputs[0]['index'], x.shape)

        # 4. 执行推理
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        # 5. Device -> Host
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)

        self.stream.synchronize()

        # 6. 解析输出
        results = []
        for out in self.outputs:
            # 动态计算当前 Batch 的实际形状
            shape = list(out['shape'])
            shape[0] = batch_size

            # 计算实际数据量
            vol = trt.volume(shape)

            # 截取并 Reshape
            data = out['host'][:vol].reshape(shape)
            results.append(data)

        return results


if __name__ == "__main__":
    # --- 全局配置 ---
    w, h = 640, 640
    # 注意：这里要填你用 best_final.onnx 转换出来的 engine 路径
    trt_engine_path = r'./onnx/best-smi.engine'
    img_path = r'./images/baofeng'
    score_threshold = 0.45

    if not os.path.exists('./results_trt'):
        os.makedirs('./results_trt')

    palette = {
        0: (0, 255, 0), 1: (0, 0, 255), 2: (255, 0, 0), 3: (255, 255, 0),
        4: (255, 0, 255), 5: (171, 130, 255), 6: (155, 211, 255), 7: (0, 255, 255)
    }
    label_map = {0: 'TA', 1: 'LV'}

    # 1. 初始化模型
    print(f"Loading TensorRT Engine: {trt_engine_path}...")
    model = TrtModel(trt_engine_path, max_batch_size=1)

    # 打印一下输入类型，确认是否为 uint8
    print(f"Model Input Dtype: {model.inputs[0]['dtype']}")  # 应该是 uint8

    pic_paths = list(paths.list_images(img_path))

    for pic_path in tqdm(pic_paths):
        basename = os.path.basename(pic_path)
        img_raw = cv2.imread(pic_path)
        if img_raw is None: continue

        # --- 获取原图尺寸 (用于坐标还原) ---
        o_H, o_W = img_raw.shape[:2]
        ratio_w = o_W / w
        ratio_h = o_H / h

        img_bak = img_raw.copy()

        # --- 2. 预处理 (严格对齐 4_onnxrun.py) ---
        # 步骤 A: 强制 Resize
        img = cv2.resize(img_raw, (w, h))

        # 步骤 B: Letterbox (虽然 r=1, 但保持逻辑一致)
        img, r, dw, dh = letterbox(img, h, w)

        # 步骤 C: HWC -> CHW
        img = np.transpose(img, (2, 0, 1))

        # 步骤 D: 增加 Batch 维度 -> (1, 3, 640, 640)
        img = np.expand_dims(img, axis=0)

        # 步骤 E: 确保连续内存和 uint8 类型
        # 这一步非常重要，因为模型输入节点是 uint8
        img = np.ascontiguousarray(img, dtype=np.uint8)

        # --- 3. 推理 ---
        # 结果是一个列表，我们的模型只有一个输出 output
        trt_outputs = model(img)

        # output shape: [Batch, TopK, 6] -> [1, 150, 6]
        output = trt_outputs[0][0]

        # --- 4. 后处理 (End-to-End) ---
        # output rows: [cx, cy, w, h, score, class_id]

        for row in output:
            score = row[4]

            # 阈值过滤 (因为已经降序排列，低于阈值直接 break)
            if score < score_threshold:
                break

            class_id = int(row[5])

            # 获取归一化坐标 (0-1)
            cx_norm, cy_norm, w_norm, h_norm = row[0], row[1], row[2], row[3]

            # --- 坐标还原 ---
            # 直接使用暴力拉伸的比例还原
            cx = cx_norm * ratio_w * w  # 等价于 cx_norm * o_W
            cy = cy_norm * ratio_h * h
            bw = w_norm * ratio_w * w
            bh = h_norm * ratio_h * h

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

            # --- 绘制 ---
            color = palette.get(class_id, (0, 255, 0))
            cv2.rectangle(img_bak, (x1, y1), (x2, y2), color, 3)

            label_text = f"{label_map.get(class_id, str(class_id))}: {score:.2f}"
            cv2.putText(img_bak, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        output_file = os.path.join('./results_trt', basename)
        cv2.imwrite(output_file, img_bak)

    print("TensorRT Inference Done.")