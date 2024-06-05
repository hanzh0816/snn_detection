from src.io.psee_loader import PSEELoader
import os
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2

import torch
import yaml
from tqdm import tqdm
from numpy.lib.recfunctions import structured_to_unstructured


def create_labels(boxes, w, h):
    """
    根据bbox构造label
    """
    # 抽出bbox数据中对应的4维
    temp = structured_to_unstructured(
        boxes[["x", "y", "w", "h"]], dtype=np.float32
    ).copy()
    torch_boxes = torch.from_numpy(temp)

    # 只需要每个目标最新的标注(可能用不到)
    _, unique_indices = np.unique(np.flip(boxes["track_id"]), return_index=True)
    unique_indices = np.flip(-(unique_indices + 1))
    torch_boxes = torch_boxes[[*unique_indices]]

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
    # valid_idx = (torch_boxes[:,2]-torch_boxes[:,0] != 0) & (torch_boxes[:,3]-torch_boxes[:,1] != 0)

    torch_boxes = torch_boxes[valid_idx, :]

    torch_labels = torch.from_numpy(boxes["class_id"]).to(torch.long)
    torch_labels = torch_labels[[*unique_indices]]
    torch_labels = torch_labels[valid_idx]

    labels = np.zeros((torch_boxes.shape[0], 5))
    labels[:, 0] = torch_labels.numpy()  # 类别标签
    labels[:, 1:] = torch_boxes.numpy()  # bbox坐标

    return {"boxes": torch_boxes, "labels": torch_labels}, labels


def create_image(events, T, img=None):
    """
    event数据到帧格式图像
    """
    width = 304
    height = 240
    # 压缩到一张图片上；
    if img is None:
        # cv2维度顺序[h,w,c]
        img = 127 * np.ones((T, width, height, 3), dtype=np.uint8)
    else:
        # if an array was already allocated just paint it grey
        img[...] = 127

    if len(events):
        for i in range(T):
            if len(events[i]):
                assert (
                    events[i]["x"].max() < width
                ), "out of bound events: x = {}, w = {}".format(
                    events[i]["x"].max(), width
                )
                assert (
                    events[i]["y"].max() < height
                ), "out of bound events: y = {}, h = {}".format(
                    events[i]["y"].max(), height
                )
            img[i, events[i]["x"], events[i]["y"], :] = (
                255 * events[i]["p"][:, None]
            )  # img默认为灰度127，有正事件则为255，无正事件则为0
    return img  # numpy.ndarray[T,W,H，C]#255 -255 每个格子


def create_sample(video:PSEELoader, boxes, sample_size, T, image_shape):
    ts = boxes["t"][0]
    video.seek_time(ts - sample_size)
    events = []
    for _ in range(T):
        events.append(video.load_delta_t(sample_size // T))

    h, w = image_shape
    targets, labels = create_labels(boxes, w, h)

    if targets["boxes"].shape[0] == 0:
        print(f"No boxes at {ts}")
        return None
    elif len(events) == 0:
        print(f"No events at {ts}")
        return None
    else:
        return (create_image(events, T), labels)


def check_path(output_path, build_path):
    if os.path.exists(output_path):
        pass
    else:
        os.mkdir(output_path)
    if os.path.exists(build_path):
        pass
    else:
        os.mkdir(build_path)
    if os.path.exists(os.path.join(build_path, "images")):
        pass
    else:
        os.mkdir(os.path.join(build_path, "images"))
    if os.path.exists(os.path.join(build_path, "labels")):
        pass
    else:
        os.mkdir(os.path.join(build_path, "labels"))


def build_dataset(input_path, output_path, sample_size, T, image_shape, task="train"):
    """
    构建yolo格式的数据集
    """
    input_path = os.path.join(input_path, task)
    build_path = os.path.join(output_path, task)
    check_path(output_path, build_path)
    # 文件夹下的所有npy文件名
    files = [
        os.path.join(input_path, time_seq_name[:-9])
        for time_seq_name in os.listdir(input_path)  # 路径下所有文件
        if time_seq_name[-3:] == "npy"
    ]

    print("Building the Dataset")
    pbar = tqdm(total=len(files), unit="File", unit_scale=True)

    save_txt_path = os.path.join(output_path, task + ".txt")

    for file_name in files:
        p = 0
        events_file = file_name + "_td.dat"
        boxes_file = file_name + "_bbox.npy"
        try:
            # video读取
            video = PSEELoader(events_file)
            # bbox读取
            boxes = np.load(boxes_file)
        except ValueError:
            continue

        # Rename 'ts' in 't' if needed (Prophesee GEN1)
        boxes.dtype.names = [
            dtype if dtype != "ts" else "t" for dtype in boxes.dtype.names
        ]  # npy文件中的name

        # 以时间戳为单位组织bbox,每个时间戳一个list
        boxes_per_ts = np.split(boxes, np.unique(boxes["t"], return_index=True)[1][1:])

        flname = file_name.split("/")[-1]  # 不包含目录的文件名

        # 逐时间戳的生成样本，每个样本为0.25s的视频分割成5帧，形成一个[T,W,H,C]格式的样本
        for b in boxes_per_ts:
            event_data, imglabels = create_sample(
                video, b, sample_size, T, image_shape
            )  # event_data:[T,H,W,C]
            svimgname = os.path.join(
                build_path, "images", "img_" + flname + str(p) + ".npy"
            )
            svlbname = os.path.join(
                build_path, "labels", "img_" + flname + str(p) + ".txt"
            )
            np.save(svimgname, event_data)

            w = event_data.shape[1]
            h = event_data.shape[2]

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

            imglabels_txt = imglabels @ scale
            if (np.array(imglabels_txt) <= 1).all() == False:
                # 判断归一化后是否有大于1的元素，保证归一化的维度正确
                print("error!")
            np.savetxt(svlbname, imglabels_txt)
            if event_data is not None:
                with open(save_txt_path, "a") as f:
                    f.write((str(svimgname) + "\n"))
            p += 1
        pbar.update(1)
    pbar.close()


sample_size = 250000
image_shape = (240, 304)
T = 5
input_path = "/data1/hzh/Gen1/Gen1/detection_dataset_duration_60s_ratio_1.0"
output_path = "/data/hzh/Gen1"
build_dataset(input_path, output_path, sample_size, T, image_shape, task="train")
