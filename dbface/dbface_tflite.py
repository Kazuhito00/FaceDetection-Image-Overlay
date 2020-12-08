#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import copy

import cv2 as cv
import numpy as np
from numpy.lib.stride_tricks import as_strided

try:
    from tflite_runtime.interpreter import Interpreter
except:
    from tensorflow.lite.python.interpreter import Interpreter


class DBFaceTflite(object):
    def __init__(
        self,
        model_path='dbface/model/dbface_keras_256x256_float16_quant_nhwc.tflite',
        num_threads=4,
    ):

        self.interpreter = Interpreter(model_path=model_path,
                                       num_threads=num_threads)

        self.interpreter.allocate_tensors()
        self.inblobs = self.interpreter.get_input_details()
        self.outblobs = self.interpreter.get_output_details()

        self.mean = np.array([0.408, 0.447, 0.47], dtype="float32")
        self.std = np.array([0.289, 0.274, 0.278], dtype="float32")

    def __call__(
        self,
        image,
        height,  # unuse
        width,  # unuse
        threshold=0.4,
        nms_iou=0.5,
    ):

        temp_image = self._pre_process(image)

        inblobs_tensor_index = self.inblobs[0]['index']
        self.interpreter.set_tensor(inblobs_tensor_index, temp_image)
        self.interpreter.invoke()

        outblobs_tensor_index_lm = self.outblobs[0]['index']
        outblobs_tensor_index_bbox = self.outblobs[1]['index']
        outblobs_tensor_index_hm = self.outblobs[2]['index']

        lm = self.interpreter.get_tensor(outblobs_tensor_index_lm)
        lm = lm[0][np.newaxis, :, :, :]  # 1,h,w,10
        bbox = self.interpreter.get_tensor(outblobs_tensor_index_bbox)
        bbox = bbox[0][np.newaxis, :, :, :]  # 1,h,w,4
        hm = self.interpreter.get_tensor(outblobs_tensor_index_hm)
        hm = hm[0][np.newaxis, :, :, :].transpose((0, 3, 1, 2))  # 1,1,h,w

        results = self._detect(
            hm=hm,
            box=bbox,
            landmark=lm,
            threshold=threshold,
            nms_iou=nms_iou,
        )

        bboxes, landmarks = self._post_process(image, results)

        return bboxes, landmarks

    def _pre_process(self, image):
        temp_iamge = copy.deepcopy(image)

        input_width = self.inblobs[0]['shape'][2]
        input_height = self.inblobs[0]['shape'][1]

        temp_iamge = cv.resize(temp_iamge, (input_width, input_height))
        temp_iamge = cv.cvtColor(temp_iamge, cv.COLOR_BGR2RGB)
        temp_iamge = temp_iamge.astype(np.float32)
        temp_iamge = ((temp_iamge / 255.0 - self.mean) / self.std).astype(
            np.float32)
        temp_iamge = temp_iamge[np.newaxis, :, :, :]

        return temp_iamge

    def _detect(self, hm, box, landmark, threshold=0.4, nms_iou=0.5):
        hm_pool = self._max_pooling(hm[0, 0, :, :], 3, 1, 1)  # 1,1,64,64
        interest_points = ((hm == hm_pool) * hm)  # screen out low-conf pixels
        flat = interest_points.ravel()  # flatten
        indices = np.argsort(flat)[::-1]  # index sort

        _, hm_width = hm.shape[1:3]
        ys = indices // hm_width
        xs = indices % hm_width

        scores = np.array([flat[idx] for idx in indices])
        box = box.reshape(box.shape[1:])  # 64,64,4
        landmark = landmark.reshape(landmark.shape[1:])  # 64,64,10

        stride = 4
        objs = []
        for cx, cy, score in zip(xs, ys, scores):
            if score < threshold:
                break

            x, y, r, b = box[cy, cx, :]
            xyrb = (np.array([cx, cy, cx, cy]) + [-x, -y, r, b]) * stride
            x5y5 = landmark[cy, cx, :]
            x5y5 = (self._exp(x5y5 * 4) + ([cx] * 5 + [cy] * 5)) * stride
            box_landmark = list(zip(x5y5[:5], x5y5[5:]))
            objs.append([xyrb, score, box_landmark])

        return self._nms(objs, iou=nms_iou)

    def _max_pooling(self, x, kernel_size, stride=1, padding=1):
        x = np.pad(x, padding, mode='constant')

        output_shape = ((x.shape[0] - kernel_size) // stride + 1,
                        (x.shape[1] - kernel_size) // stride + 1)

        kernel = (kernel_size, kernel_size)

        x_w = as_strided(
            x,
            shape=output_shape + kernel,
            strides=(stride * x.strides[0], stride * x.strides[1]) + x.strides)
        x_w = x_w.reshape(-1, *kernel)

        return x_w.max(axis=(1, 2)).reshape(output_shape)

    def _exp(self, v):
        if isinstance(v, tuple) or isinstance(v, list):
            return [self._exp(item) for item in v]
        elif isinstance(v, np.ndarray):
            return np.array([self._exp(item) for item in v], v.dtype)

        gate = 1
        base = np.exp(1)
        if abs(v) < gate:
            return v * base
        if v > 0:
            return np.exp(v)
        else:
            return -np.exp(-v)

    def _nms(self, objs, iou=0.5):
        if objs is None or len(objs) <= 1:
            return objs

        objs = sorted(objs, key=lambda obj: obj[1], reverse=True)

        keep = []
        flags = [0] * len(objs)
        for index, obj in enumerate(objs):
            if flags[index] != 0:
                continue

            keep.append(obj)

            for j in range(index + 1, len(objs)):
                if flags[j] == 0 and self._iou(obj[0], objs[j][0]) > iou:
                    flags[j] = 1

        return keep

    def _iou(self, rec1, rec2):
        cx1, cy1, cx2, cy2 = rec1
        gx1, gy1, gx2, gy2 = rec2

        x1 = max(cx1, gx1)
        y1 = max(cy1, gy1)
        x2 = min(cx2, gx2)
        y2 = min(cy2, gy2)

        w = max(0, x2 - x1 + 1)
        h = max(0, y2 - y1 + 1)
        area = w * h

        S_rec1 = (cx2 - cx1 + 1) * (cy2 - cy1 + 1)
        S_rec2 = (gx2 - gx1 + 1) * (gy2 - gy1 + 1)
        iou = area / (S_rec1 + S_rec2 - area)

        return iou

    def _post_process(self, image, results):
        image_width = image.shape[1]
        image_height = image.shape[0]
        input_width = self.inblobs[0]['shape'][2]
        input_height = self.inblobs[0]['shape'][1]

        scale_w = image_width / input_width
        scale_h = image_height / input_height

        bboxes = []
        landmarks = []

        for result in results:
            score = result[1]

            bbox_x1 = int(result[0][0] * scale_w)
            bbox_y1 = int(result[0][1] * scale_h)
            bbox_x2 = int(result[0][2] * scale_w)
            bbox_y2 = int(result[0][3] * scale_h)

            bboxes.append([bbox_x1, bbox_y1, bbox_x2, bbox_y2, score])

            landmark = []
            for landmark_point in result[2]:
                landmark_x, landmark_y = landmark_point[:2]
                landmark_x = int(landmark_x * scale_w)
                landmark_y = int(landmark_y * scale_h)

                landmark.append([landmark_x, landmark_y])

            landmarks.append(landmark)

        return bboxes, landmarks


if __name__ == '__main__':
    pass
