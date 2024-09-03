from curses import raw
import os
import torch
import cv2
import yaml
import json
import time
from bisect import bisect_left
from dv import AedatFile
import multiprocessing as mp

import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

from matplotlib import pyplot as plt


def create_labels(data):

    w, h = data["imageWidth"], data["imageHeight"]

    temp = []
    labels = []
    for shape in data["shapes"]:
        bbox = np.array(shape["points"]).flatten()
        bbox_w = bbox[2] - bbox[0]
        bbox_h = bbox[3] - bbox[1]
        bbox[2:] = bbox_w, bbox_h

        labels.append(int(shape["label"]))

        temp.append(bbox)
    labels = np.array(labels)
    temp = np.array(temp)

    torch_boxes = torch.from_numpy(temp)
    # 先约束到图像范围内
    torch_boxes[:, 0::2].clamp_(min=0, max=w)  # 第0和第2个元素
    torch_boxes[:, 1::2].clamp_(min=0, max=h)
    # 左上角坐标变为中心坐标
    torch_boxes[:, 0] = torch_boxes[:, 0] + torch_boxes[:, 2] / 2
    torch_boxes[:, 1] = torch_boxes[:, 1] + torch_boxes[:, 3] / 2
    # double check
    torch_boxes[:, 0::2].clamp_(min=0, max=w)  # 第0和第2个元素
    torch_boxes[:, 1::2].clamp_(min=0, max=h)

    # valid idx = width and height of GT bbox aren't 0
    valid_idx = (torch_boxes[:, 2] != 0) & (torch_boxes[:, 3] != 0)

    torch_boxes = torch_boxes[valid_idx, :]

    torch_labels = torch.from_numpy(labels).to(torch.long)
    torch_labels = torch_labels[valid_idx]

    labels = np.zeros((torch_boxes.shape[0], 5))
    labels[:, 0] = torch_labels.numpy()  # 类别标签
    labels[:, 1:] = torch_boxes.numpy()  # bbox坐标
    scale = np.array(
        [
            [1, 0, 0, 0, 0],
            [0, 1 / w, 0, 0, 0],
            [0, 0, 1 / h, 0, 0],
            [0, 0, 0, 1 / w, 0],
            [0, 0, 0, 0, 1 / h],
        ],
        dtype=float,
    )
    labels = labels @ scale
    assert (np.array(labels[:, 1:]) <= 1).all() == True, "label normalization errors"

    return {"boxes": torch_boxes, "labels": torch_labels, "w": w, "h": h}, labels


def create_image(events, width, height, T):
    """
    event数据到帧格式图像
    """
    # 压缩到一张图片上；
    # cv2维度顺序[h,w,c]
    img = 127 * np.ones((T, width, height, 3), dtype=np.uint8)

    if len(events):
        for i in range(T):
            if len(events[i]):
                events_list = np.array(list(map(list, events[i])))
                x = events_list[:, 1]
                y = events_list[:, 2]
                p = events_list[:, 3]
                assert x.max() < width, "out of bound events: x = {}, w = {}".format(x.max(), width)
                assert y.max() < height, "out of bound events: y = {}, h = {}".format(
                    y.max(), height
                )

                img[i, x, y, :] = (
                    255 * p[:, None]
                )  # img默认为灰度127，有正事件则为255，无正事件则为0
    return img  # numpy.ndarray[T,W,H，C]#255 -255 每个格子


def display_image(img):
    """
    Display T images on a canvas.

    Parameters:
    images (numpy.ndarray): An array of shape (T, width, height, 3).
    """
    T, width, height, _ = img.shape
    fig, axes = plt.subplots(1, T, figsize=(15, 5))

    if T == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        # Transpose each image from (width, height, 3) to (height, width, 3)
        hwc_image = img[i].transpose(1, 0, 2)
        ax.imshow(hwc_image)
        ax.axis("off")
        ax.set_title(f"Frame {i+1}")

    plt.tight_layout()
    plt.show()


def create_one_sample(
    input_aedat_file, raw_path, build_path, prefix, scene, sample_size, T, task="train"
):
    label_path = os.path.join(raw_path, "annotations", task, scene, prefix)
    print(f"processing prefix :{input_aedat_file}")
    frame_timestamps = []
    try:
        with AedatFile(input_aedat_file) as f:
            for frame in f["frames"]:
                aps_timestamp = frame.timestamp
                frame_timestamps.append(aps_timestamp)

            events = np.hstack([event for event in f["events"].numpy()])
            event_timestamps = [event[0] for event in events]
            start_time = frame_timestamps[0]

            for frame_id, t in enumerate(frame_timestamps):
                # 样本不足
                if t - start_time < sample_size:
                    continue
                events_sample = []

                # label id = frame id + 1
                label_idx = frame_id + 1
                label_json_file = os.path.join(label_path, f"{label_idx}.json")
                if not os.path.exists(label_json_file):
                    continue

                save_image_path = os.path.join(
                    build_path, "images", "img_" + prefix + "_" + str(frame_id) + ".npy"
                )
                save_label_path = os.path.join(
                    build_path, "labels", "img_" + prefix + "_" + str(frame_id) + ".txt"
                )
                if os.path.exists(save_image_path):
                    continue

                try:
                    with open(label_json_file, "r") as f:
                        data = json.load(f)
                except:
                    continue
                targets, labels = create_labels(data)

                for i in range(T):
                    start_delta_T = t - sample_size + i * (sample_size // T)
                    end_delta_T = start_delta_T + sample_size // T
                    if not start_delta_T in event_timestamps:
                        event_start_index = bisect_left(event_timestamps, start_delta_T)
                    else:
                        event_start_index = event_timestamps.index(start_delta_T)

                    if not end_delta_T in event_timestamps:
                        event_end_index = bisect_left(event_timestamps, end_delta_T)
                    else:
                        event_end_index = event_timestamps.index(end_delta_T)

                    corresponding_events = events[event_start_index:event_end_index]
                    events_sample.append(corresponding_events)

                img_data = create_image(events_sample, targets["w"], targets["h"], T)
                display_image(img_data)
                input("请输入指令以继续: ")
                np.save(save_image_path, img_data)
                np.savetxt(save_label_path, labels)

                start_time = t
    except:
        print(f"Warning: can not process sample : {input_aedat_file}")


def create_one_sample_wrapper(args):
    return create_one_sample(*args)


def build_dataset(raw_path, output_path, sample_size, T, task="train"):
    print(task)
    input_path = os.path.join(raw_path, "raw", task)
    output_path = os.path.join(output_path, task)

    if not os.path.exists(os.path.join(output_path, "images")):
        os.makedirs(os.path.join(output_path, "images"))
    if not os.path.exists(os.path.join(output_path, "labels")):
        os.makedirs(os.path.join(output_path, "labels"))

    scenes = os.listdir(input_path)
    work_args = []
    for scene in scenes:
        input_files = os.listdir(os.path.join(input_path, scene))
        for fname in input_files:
            prefix = fname[:-7]
            input_aedat_file = os.path.join(input_path, scene, fname)
            create_one_sample(
                input_aedat_file,
                raw_path,
                output_path,
                prefix,
                scene,
                sample_size,
                T,
                task,
            )

            work_args.append(
                (
                    input_aedat_file,
                    raw_path,
                    output_path,
                    prefix,
                    scene,
                    sample_size,
                    T,
                    task,
                )
            )

    with mp.Pool(processes=8) as pool:
        pool.map(create_one_sample_wrapper, work_args)


raw_path = "/data1/hzh/pku/PKU_DAVIS_SOD/"
output_path = "/data1/hzh/pku/test"
sample_size = 60000
T = 4

build_dataset(
    raw_path=raw_path,
    output_path=output_path,
    sample_size=sample_size,
    T=T,
    task="train",
)
