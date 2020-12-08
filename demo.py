#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
[summary]
  CenterFaceを用いて顔検出を行い、顔上に画像を重畳表示するデモ
[description]
  -
"""
import glob
import argparse
import cv2 as cv

from centerface import CenterFace
from dbface import DBFaceTflite

from utils import CvOverlayImage


def get_args():
    """
    [summary]
        引数解析
    Parameters
    ----------
    None
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("--device",
                        type=int,
                        help='camera device number',
                        default=0)
    parser.add_argument("--width", help='capture width', type=int, default=960)
    parser.add_argument("--height",
                        help='capture height',
                        type=int,
                        default=540)
    parser.add_argument("--ceil", type=int, default=50)
    parser.add_argument("--image_ratio", type=float, default=1.2)
    parser.add_argument("--x_offset", type=int, default=0)
    parser.add_argument("--y_offset", type=int, default=-30)
    parser.add_argument("--use_model", type=str, default='centerface')

    args = parser.parse_args()

    return args


def main():
    # 引数解析 #################################################################
    args = get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height
    ceil_num = args.ceil
    image_ratio = args.image_ratio
    x_offset = args.x_offset
    y_offset = args.y_offset
    use_model = args.use_model

    # カメラ準備 ###############################################################
    cap = cv.VideoCapture(cap_device)

    # 重畳画像準備 #############################################################
    image_pathlist = sorted(glob.glob('image/*.png'))
    images = []
    for image_path in image_pathlist:
        images.append(cv.imread(image_path, cv.IMREAD_UNCHANGED))

    animation_counter = 0

    # facedetector準備 #########################################################
    if use_model == 'centerface':
        facedetector = CenterFace()
    elif use_model == 'dbface':
        facedetector = DBFaceTflite()

    while True:
        # カメラキャプチャ #####################################################
        ret, frame = cap.read()
        if not ret:
            break
        resize_frame = cv.resize(frame, (int(cap_width), int(cap_height)))
        frame_height, frame_width = resize_frame.shape[:2]

        # 顔検出 ##############################################################
        dets, lms = facedetector(resize_frame,
                                 frame_height,
                                 frame_width,
                                 threshold=0.35)

        # デバッグ表示 ########################################################
        # バウンディングボックス
        for det in dets:
            bbox, _ = det[:4], det[4]  # BBox, Score
            x1, y1 = int(bbox[0]), int(bbox[1])
            x2, y2 = int(bbox[2]), int(bbox[3])

            # cv.rectangle(resize_frame, (x1, y1), (x2, y2), (2, 255, 0), 1)

            # 顔の立幅に合わせて重畳画像をリサイズ
            image_height, image_width = images[0].shape[:2]
            resize_ratio = (y2 - y1) / image_height
            resize_image_height = int(image_height * resize_ratio)
            resize_image_width = int(image_width * resize_ratio)

            resize_image_height = int(
                (resize_image_height + (ceil_num - 1)) / ceil_num * ceil_num)
            resize_image_width = int(
                (resize_image_width + (ceil_num - 1)) / ceil_num * ceil_num)
            resize_image_height = int(resize_image_height * image_ratio)
            resize_image_width = int(resize_image_width * image_ratio)

            resize_image = cv.resize(images[animation_counter],
                                     (resize_image_width, resize_image_height))

            # 画像描画
            overlay_x = int((x2 + x1) / 2) - int(resize_image_width / 2)
            overlay_y = int((y2 + y1) / 2) - int(resize_image_height / 2)
            resize_frame = CvOverlayImage.overlay(
                resize_frame, resize_image,
                (overlay_x + x_offset, overlay_y + y_offset))

        # ランドマーク
        # for lm in lms:
        #     for lm_point in lm:
        #         cv.circle(resize_frame, (
        #             lm_point[0],
        #             lm_point[1],
        #         ), 3, (0, 0, 255), -1)

        animation_counter += 1
        if animation_counter >= len(images):
            animation_counter = 0

        cv.imshow('Debug', resize_frame)
        key = cv.waitKey(1)
        if key == 27:  # ESC
            break
    cap.release()


if __name__ == '__main__':
    main()
