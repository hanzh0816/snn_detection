import argparse
from copy import deepcopy
import json
from pathlib import Path
from functools import lru_cache
from turtle import st

import cv2
import math
import h5py
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import random
import shutil

CATEGORIES = [
    {"id": 0, "name": "pedestrian"},
    {"id": 1, "name": "rider"},
    {"id": 2, "name": "car"},
    {"id": 3, "name": "bus"},
    {"id": 4, "name": "truck"},
    {"id": 5, "name": "bicycle"},
    {"id": 6, "name": "motorcycle"},
    {"id": 7, "name": "train"},
]


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument(
        "--root-dir", type=Path, help="root directory of DSEC dataset", default="/data2/hzh/DSEC/"
    )
    parse.add_argument(
        "--output-dir",
        type=Path,
        help="output directory of converted dataset",
        default="/data2/hzh/DSEC-COCO/",
    )
    parse.add_argument("--mode", type=str, help="mode of conversion", default="train")

    args = parse.parse_args()
    return args


def extract_mini_dataset(root_dir: Path, output_dir: Path, sample_ratio):
    mode_list = ["train", "test"]
    for mode in mode_list:
        annotations_path = root_dir / "annotations" / f"{mode}.json"
        with open(annotations_path, "r") as f:
            coco_data = json.load(f)
        samples_path = coco_data["images"]
        # 计算抽样数量
        num_samples = int(len(samples_path) * sample_ratio)

        # 随机抽取样本
        samples = random.sample(samples_path, num_samples)
        samples_id = set(sample["id"] for sample in samples)

        # 筛选对应的标注
        sampled_annotations = [
            ann for ann in coco_data["annotations"] if ann["image_id"] in samples_id
        ]
        # 构建新的标注文件
        sampled_coco_data = deepcopy(coco_data)
        sampled_coco_data["images"] = samples
        sampled_coco_data["annotations"] = sampled_annotations

        # 保存新的标注文件到输出目录
        output_annotations_path = output_dir / "annotations" / f"{mode}.json"
        output_annotations_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_annotations_path, "w") as f:
            json.dump(sampled_coco_data, f)

        # 复制抽取的图像到新文件夹
        for sample in samples:
            src_img_path = root_dir / mode / sample["file_name"] / f"{sample['file_name']}.png"
            dst_img_path = output_dir / mode / sample["file_name"] / f"{sample['file_name']}.png"
            src_event_path = root_dir / mode / sample["file_name"] / f"{sample['file_name']}.npy"
            dst_event_path = output_dir / mode / sample["file_name"] / f"{sample['file_name']}.npy"

            dst_img_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src_img_path, dst_img_path)
            shutil.copyfile(src_event_path, dst_event_path)

        print(f"抽取完成：{num_samples} 张图像已保存到 {output_dir}")
        print(f"新标注文件已保存到 {output_annotations_path}")


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


class BaseDirectory:
    def __init__(self, root):
        self.root = root


class DSECDirectory:
    def __init__(self, root):
        self.root = root
        self.images = ImageDirectory(root / "images")
        self.events = EventDirectory(root / "events")
        self.tracks = TracksDirectory(root / "object_detections")


class ImageDirectory(BaseDirectory):
    @property
    @lru_cache(maxsize=1)
    def timestamps(self):
        return np.genfromtxt(self.root / "timestamps.txt", dtype="int64")

    @property
    @lru_cache(maxsize=1)
    def image_files_rectified(self):
        return sorted(list((self.root / "left/rectified").glob("*.png")))

    @property
    @lru_cache(maxsize=1)
    def image_files_distorted(self):
        return sorted(list((self.root / "left/distorted").glob("*.png")))


class EventDirectory(BaseDirectory):
    @property
    @lru_cache(maxsize=1)
    def event_file(self):
        return self.root / "left/events.h5"


class TracksDirectory(BaseDirectory):
    @property
    @lru_cache(maxsize=1)
    def tracks(self):
        return np.load(self.root / "left/tracks.npy")


class DSECDataset:
    def __init__(self, root_dir, output_dir, mode):
        self.root_dir = root_dir / mode
        self.output_dir = output_dir / mode
        self.mode = mode
        self.annotations_path = output_dir / "annotations" / f"{self.mode}.json"
        self.annotations_path.parent.mkdir(parents=True, exist_ok=True)
        self.height = 480
        self.width = 640

        self.directories = dict()  # 保存子序列对象，key:子序列名, value:子序列对象

        # 每个元素是记录每个子序列中每个图像对应的track_idxs
        # 即当前图像对应的标注在tracks中的索引范围
        self.img_idx_track_idxs = dict()

        # 所有子序列的目录路径（按开始时间排序）
        self.subsequence_directories = list(self.root_dir.glob("*/"))
        self.subsequence_directories = sorted(
            self.subsequence_directories, key=self.first_time_from_subsequence
        )
        for f in self.subsequence_directories:
            directory = DSECDirectory(f)
            self.directories[f.name] = directory

            self.img_idx_track_idxs[f.name] = self.compute_img_idx_to_track_idx(
                directory.tracks.tracks["t"], directory.images.timestamps
            )

    def first_time_from_subsequence(self, subsequence):
        return np.genfromtxt(subsequence / "images/timestamps.txt", dtype="int64")[0]

    def convert_to_coco(self):
        sample_id = 0
        ann_id = 0
        imags = []
        annotations = []
        categories = CATEGORIES

        for f in tqdm(self.subsequence_directories, desc="Processing directories"):
            img_idx_to_track_idxs = self.img_idx_track_idxs[f.name]
            directory = self.directories[f.name]
            imgs_path = directory.images.image_files_distorted

            for img_idx in tqdm(
                range(len(imgs_path) - 1), desc=f"Processing images in {f.name}", leave=False
            ):
                sample_fname = f"{f.name}_{imgs_path[img_idx].stem}"
                # 读取图像
                img = cv2.imread(imgs_path[img_idx])
                # 获取事件对应前后图像索引窗口
                img_idx_start, img_idx_end = self.get_index_window(img_idx, len(imgs_path))
                # 获取索引对应图像时间戳（us）
                start_timestamp, end_timestamp = directory.images.timestamps[
                    [img_idx_start, img_idx_end]
                ]
                # 跳过无事件帧
                if start_timestamp == end_timestamp:
                    continue
                # 获取raw事件流
                events = self.extract_from_h5_by_timewindow(
                    directory.events.event_file, start_timestamp, end_timestamp
                )
                # 获取numpy格式事件数据
                event_npy = self.process_events_to_npy(events)
                # 获取raw标注信息
                track_start_idx, track_end_idx = img_idx_to_track_idxs[img_idx_end]  # 跳过无标注帧
                if track_start_idx == track_end_idx:
                    continue
                tracks = directory.tracks.tracks[track_start_idx:track_end_idx]
                anns, ann_id = self.process_tracks_to_dict(tracks, ann_id, sample_id)

                # 添加到annotations.json的内容
                annotations.extend(anns)
                imags.append(
                    {
                        "id": sample_id,
                        "width": self.width,
                        "height": self.height,
                        "file_name": sample_fname,
                    }
                )
                self.save_sample(sample_fname, img, event_npy)
                sample_id += 1

        ann_result = {"images": imags, "annotations": annotations, "categories": categories}
        with open(self.annotations_path, "w") as f:
            json.dump(ann_result, f, indent=4)

    def save_sample(self, sample_fname, img, events):
        save_path = self.output_dir / sample_fname  # type: Path
        save_path.mkdir(parents=True, exist_ok=True)
        np.save(save_path / f"{sample_fname}.npy", events)
        cv2.imwrite(save_path / f"{sample_fname}.png", img)

    def get_index_window(self, index, num_idx, sync="back"):
        if sync == "front":
            assert 0 < index < num_idx
            i_0 = index - 1
            i_1 = index
        else:
            assert 0 <= index < num_idx - 1
            i_0 = index
            i_1 = index + 1

        return i_0, i_1

    def extract_from_h5_by_timewindow(self, h5file, t_min_us: int, t_max_us: int):
        with h5py.File(str(h5file), "r") as h5f:
            ms2idx = np.asarray(h5f["ms_to_idx"], dtype="int64")
            t_offset = h5f["t_offset"][()]

            events = h5f["events"]
            t = events["t"]

            t_ev_start_us = t_min_us - t_offset
            # assert t_ev_start_us >= t[0], (t_ev_start_us, t[0])
            t_ev_start_ms = t_ev_start_us // 1000
            ms2idx_start_idx = t_ev_start_ms
            ev_start_idx = ms2idx[ms2idx_start_idx]

            t_ev_end_us = t_max_us - t_offset
            assert t_ev_end_us <= t[-1], (t_ev_end_us, t[-1])
            t_ev_end_ms = math.floor(t_ev_end_us / 1000)
            ms2idx_end_idx = t_ev_end_ms
            ev_end_idx = ms2idx[ms2idx_end_idx]

            return self._extract_from_h5_by_index(h5f, ev_start_idx, ev_end_idx)

    def _extract_from_h5_by_index(self, filehandle, ev_start_idx: int, ev_end_idx: int):
        events = filehandle["events"]
        x = events["x"]
        y = events["y"]
        p = events["p"]
        t = events["t"]

        x_new = x[ev_start_idx:ev_end_idx]
        y_new = y[ev_start_idx:ev_end_idx]
        p_new = p[ev_start_idx:ev_end_idx]
        t_new = t[ev_start_idx:ev_end_idx].astype("int64") + filehandle["t_offset"][()]

        output = {
            "p": p_new,
            "t": t_new,
            "x": x_new,
            "y": y_new,
        }
        return output

    @staticmethod
    def compute_img_idx_to_track_idx(t_track, t_image):
        x, counts = np.unique(t_track, return_counts=True)
        i, j = (x.reshape((-1, 1)) == t_image.reshape((1, -1))).nonzero()
        deltas = np.zeros_like(t_image)

        deltas[j] = counts[i]

        idx = np.concatenate([np.array([0]), deltas]).cumsum()
        return np.stack([idx[:-1], idx[1:]], axis=-1).astype("uint64")

    @staticmethod
    def process_events_to_npy(events, T=4, width=640, height=480):

        # 按[T,H,W,C]的顺序写入
        event_npy = 127 * np.ones((T, height, width, 3), dtype=np.uint8)
        p, t, x, y = events["p"], events["t"], events["x"], events["y"]
        p, t, x, y = (
            np.array_split(p, T),
            np.array_split(t, T),
            np.array_split(x, T),
            np.array_split(y, T),
        )
        for i in range(T):
            assert (
                x[i].max() < width and x[i].min() >= 0
            ), "out of bound events: x = {}, w = {}".format(x[i].max(), width)

            assert (
                y[i].max() < height and y[i].min() >= 0
            ), "out of bound events: y = {}, h = {}".format(y[i].max(), height)

            event_npy[i, y[i], x[i], :] = (
                255 * p[i][:, None]
            )  # img默认为灰度127，有正事件则为255，无正事件则为0
        return event_npy

    @staticmethod
    def process_tracks_to_dict(tracks, ann_id, img_id):
        annotations = []
        for track in tracks:
            _, x, y, h, w, class_id, _, _ = track
            if x < 0:
                x = 0
            if y < 0:
                y = 0
            assert all([x >= 0, y >= 0, h > 0, w > 0]), "invalid track: {}".format(track)
            annotations.append(
                {
                    "id": ann_id,
                    "category_id": int(class_id),
                    "image_id": img_id,
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "area": int(w * h),
                }
            )
            ann_id += 1
        return annotations, ann_id


def main():
    args = parse_args()
    dataset = DSECDataset(args.root_dir, args.output_dir, args.mode)
    dataset.convert_to_coco()


if __name__ == "__main__":
    main()
