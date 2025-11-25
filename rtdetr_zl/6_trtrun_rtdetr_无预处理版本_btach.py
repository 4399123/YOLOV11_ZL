# encoding=gbk
import tensorrt as trt
import numpy as np
import os
import cv2
import pycuda.driver as cuda
import pycuda.autoinit
from imutils import paths
from tqdm import tqdm
from letterBOX import letterbox


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

            # --- 核心修复 ---
            # 不管 Engine 返回的是 -1 还是 1，只要是第0维（Batch维），
            # 我们都强制按 max_batch_size 分配显存。
            # 这样即使 Engine 被误识别为静态 Batch=1，显存也是够用的（宁大勿小）。
            if dims[0] == -1 or dims[0] == 1:
                dims[0] = self.max_batch_size

            # 打印一下分配的形状，方便调试
            # print(f"Binding {i} allocation shape: {dims}")

            size = trt.volume(dims)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))

            # 分配 Host 和 Device 内存
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
        batch_size = x.shape[0]
        input_info = self.inputs[0]

        # 展平数据
        x_flat = x.ravel().astype(input_info['dtype'])

        # 拷贝到 Host (现在显存足够大了，不会报错)
        np.copyto(input_info['host'][:x_flat.size], x_flat)

        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)

        # 设置动态 Input Shape
        # 如果 Engine 是纯静态 Batch=1 的，这一步可能会被忽略，但不会报错
        self.context.set_binding_shape(self.inputs[0]['index'], x.shape)

        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)

        self.stream.synchronize()

        results = []
        for out in self.outputs:
            shape = list(out['shape'])
            shape[0] = batch_size
            vol = trt.volume(shape)
            data = out['host'][:vol].reshape(shape)
            results.append(data)
        return results


if __name__ == "__main__":
    # --- 配置 ---
    w, h = 640, 640
    trt_engine_path = r'./onnx/best-smi.engine'  # 确保路径正确
    img_path = r'./images/baofeng'
    score_threshold = 0.45

    # 设置 Batch Size
    BATCH_SIZE = 2

    if not os.path.exists('./results_trt_batch'):
        os.makedirs('./results_trt_batch')

    palette = {0: (0, 255, 0), 1: (0, 0, 255), 2: (255, 0, 0)}
    label_map = {0: 'TA', 1: 'LV'}

    print(f"Loading Engine: {trt_engine_path}...")
    # 初始化模型，分配足够 Batch=2 的显存
    model = TrtModel(trt_engine_path, max_batch_size=BATCH_SIZE)

    all_pic_paths = list(paths.list_images(img_path))

    # 简单的边界检查，防止图片不足导致报错
    if len(all_pic_paths) == 0:
        print("No images found!")
        exit()

    for i in tqdm(range(0, len(all_pic_paths), BATCH_SIZE)):
        batch_paths = all_pic_paths[i: i + BATCH_SIZE]
        batch_imgs = []
        batch_info = []

        for pic_path in batch_paths:
            img_raw = cv2.imread(pic_path)
            if img_raw is None: continue

            basename = os.path.basename(pic_path)
            o_H, o_W = img_raw.shape[:2]

            info = {
                "raw": img_raw.copy(),
                "ratio_w": o_W / w,
                "ratio_h": o_H / h,
                "name": basename
            }
            batch_info.append(info)

            img = cv2.resize(img_raw, (w, h))
            img, r, dw, dh = letterbox(img, h, w)
            img = np.transpose(img, (2, 0, 1))
            batch_imgs.append(img)

        if not batch_imgs: continue

        batch_input = np.stack(batch_imgs, axis=0)
        batch_input = np.ascontiguousarray(batch_input, dtype=np.uint8)

        # 推理
        try:
            trt_outputs = model(batch_input)
            batch_output = trt_outputs[0]
        except Exception as e:
            print(f"\n[Error] Inference failed: {e}")
            print("Tip: If this is a status 4 error, your Engine might be static Batch=1.")
            continue

        # 后处理
        for b_idx in range(len(batch_info)):
            output = batch_output[b_idx]
            info = batch_info[b_idx]
            img_bak = info["raw"]
            ratio_w = info["ratio_w"]
            ratio_h = info["ratio_h"]

            for row in output:
                score = row[4]
                if score < score_threshold: break

                class_id = int(row[5])
                cx_norm, cy_norm, w_norm, h_norm = row[0], row[1], row[2], row[3]

                cx = cx_norm * ratio_w * w
                cy = cy_norm * ratio_h * h
                bw = w_norm * ratio_w * w
                bh = h_norm * ratio_h * h

                x1 = int(cx - bw / 2)
                y1 = int(cy - bh / 2)
                x2 = int(cx + bw / 2)
                y2 = int(cy + bh / 2)

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(info["raw"].shape[1], x2), min(info["raw"].shape[0], y2)

                color = palette.get(class_id, (0, 255, 0))
                cv2.rectangle(img_bak, (x1, y1), (x2, y2), color, 3)
                label_text = f"{label_map.get(class_id, str(class_id))}: {score:.2f}"
                cv2.putText(img_bak, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            cv2.imwrite(os.path.join('./results_trt_batch', info["name"]), img_bak)

    print("Batch Inference Done.")