#encoding=gbk
import tensorrt as trt
import numpy as np
import os
import cv2
import pycuda.driver as cuda
import pycuda.autoinit
from imutils import paths
from tqdm import tqdm


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class TrtModel:

    def __init__(self, engine_path, max_batch_size=1, dtype=np.float32):

        self.engine_path = engine_path
        self.dtype = dtype
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.load_engine(self.runtime, self.engine_path)
        self.max_batch_size = max_batch_size
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers()
        self.context = self.engine.create_execution_context()

    @staticmethod
    def load_engine(trt_runtime, engine_path):
        trt.init_libnvinfer_plugins(None, "")
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        engine = trt_runtime.deserialize_cuda_engine(engine_data)
        return engine

    def allocate_buffers(self):

        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        for binding in self.engine:
            # size = trt.volume(self.engine.get_binding_shape(binding)) * self.max_batch_size
            #*******
            ssize = self.engine.get_binding_shape(binding)
            ssize[0]=self.max_batch_size
            size=trt.volume(ssize)
            #*******
            host_mem = cuda.pagelocked_empty(size, self.dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))

        return inputs, outputs, bindings, stream

    def __call__(self, x: np.ndarray, batch_size=2):

        x = x.astype(self.dtype)

        np.copyto(self.inputs[0].host, x.ravel())

        for inp in self.inputs:
            cuda.memcpy_htod_async(inp.device, inp.host, self.stream)

        #**********
        origin_inputshape=self.engine.get_binding_shape(0)
        origin_inputshape[0]=batch_size
        self.context.set_binding_shape(0,(origin_inputshape))
        #**********

        self.context.execute_async(batch_size=batch_size, bindings=self.bindings, stream_handle=self.stream.handle)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream)

        self.stream.synchronize()

        return [out.host.reshape(batch_size, -1) for out in self.outputs]


if __name__ == "__main__":

    w, h = 640, 640
    path=r'./images/'
    trt_engine_path = r'./pt/catdog/v8_n.engine'

    if not os.path.exists('./results'):
        os.makedirs('./results')

    palette = {0: (0, 255, 0),
               1: (0, 0, 255),
               2: (255, 0, 0),
               3: (255, 255, 0),
               4: (255, 0, 255),
               5: (171, 130, 255),
               6: (155, 211, 255),
               7: (0, 255, 255)}

    label = {0: 'cat',
             1: 'dog',
             2: 'eagle',
             3: 'elephant'}
    score_threshold = 0.5
    nms_threshold = 0.45


    model = TrtModel(trt_engine_path)     #构建TRT模型，这部分tensorrt有对应接口

    pic_paths = list(paths.list_images(path))
    for pic_path in tqdm(pic_paths):
        basename = os.path.basename(pic_path)
        img = cv2.imread(pic_path)
        H, W = img.shape[0], img.shape[1]
        h_ratio = H / h
        w_ratio = W / w
        imgbak = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (w, h)).astype(np.float32)
        img = img / 255.0
        img = np.array([np.transpose(img, (2, 0, 1))])

        out = model(img, 1)
        out=np.reshape(out,(1,4+len(label),-1))
        outputs = np.array([cv2.transpose(out[0])])
        rows = outputs.shape[1]

        boxes = []
        scores = []
        class_ids = []

        # Iterate through output to collect bounding boxes, confidence scores, and class IDs
        for i in range(rows):
            classes_scores = outputs[0][i][4:]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
            if maxScore >= score_threshold:
                box = [
                    float(outputs[0][i][0] - (0.5 * outputs[0][i][2])),  # x center - width/2 = left x
                    float(outputs[0][i][1] - (0.5 * outputs[0][i][3])),  # y center - height/2 = top y
                    float(outputs[0][i][2]),  # width
                    float(outputs[0][i][3]),  # height
                ]
                boxes.append(box)
                scores.append(maxScore)
                class_ids.append(maxClassIndex)

        # Apply NMS (Non-maximum suppression)
        result_boxes = cv2.dnn.NMSBoxes(bboxes=boxes, scores=scores, score_threshold=score_threshold,
                                        nms_threshold=nms_threshold)
        for i in range(len(result_boxes)):
            index = result_boxes[i][0]
            box = boxes[index]
            id = class_ids[index]
            score = scores[index]
            x1 = int(box[0] * w_ratio)
            y1 = int(box[1] * h_ratio)
            x2 = int((box[0] + box[2]) * w_ratio)
            y2 = int((box[1] + box[3]) * h_ratio)
            color = palette[id]
            cv2.rectangle(imgbak, (x1, y1), (x2, y2), color, 2)
        cv2.imwrite('./results/{}_res.jpg'.format(basename), imgbak)






