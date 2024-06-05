import json
import os
import shutil
import numpy as np
import cv2


info = {"year": 2024, "version": "1.0", "date_created": 2024 - 3 - 29}

licenses = {
    "id": 1,
    "name": "null",
    "url": "null",
}

categories = [
    {
        "id": 0,
        "name": "cars",
        "supercategory": "lines",
    },
    {
        "id": 1,
        "name": "pedestrians",
        "supercategory": "lines",
    },
]


train_data = {
    "info": info,
    "licenses": licenses,
    "categories": categories,
    "images": [],
    "annotations": [],
}
test_data = {
    "info": info,
    "licenses": licenses,
    "categories": categories,
    "images": [],
    "annotations": [],
}
valid_data = {
    "info": info,
    "licenses": licenses,
    "categories": categories,
    "images": [],
    "annotations": [],
}


def gen1_yolo_covert_coco_format(image_path, label_path):
    images = []
    annotations = []
    for index, img_file in enumerate(os.listdir(image_path)):
        if img_file.endswith(".npy"):
            image_info = {}
            img = np.load(os.path.join(image_path, img_file))  # T,W,H,C
            width, height = img.shape[1:3]
            image_info["id"] = index
            image_info["file_name"] = img_file
            image_info["width"], image_info["height"] = width, height
        else:
            continue
        if image_info != {}:
            images.append(image_info)
        # 处理label信息-------
        label_file = os.path.join(label_path, img_file.replace(".npy", ".txt"))
        with open(label_file, "r") as f:
            for idx, line in enumerate(f.readlines()):
                info_annotation = {}
                class_num, xs, ys, ws, hs = line.strip().split(" ")
                class_id, xc, yc, w, h = (
                    round(float(class_num)),
                    float(xs),
                    float(ys),
                    float(ws),
                    float(hs),
                )
                xmin = (xc - w / 2) * width
                ymin = (yc - h / 2) * height
                xmax = (xc + w / 2) * width
                ymax = (yc + h / 2) * height
                bbox_w = int(width * w)
                bbox_h = int(height * h)

                info_annotation["category_id"] = class_id  # 类别的id
                info_annotation["bbox"] = [xmin, ymin, bbox_w, bbox_h]  ## bbox的坐标
                info_annotation["area"] = bbox_h * bbox_w  ###area
                info_annotation["image_id"] = index  # bbox的id
                info_annotation["id"] = index * 100 + idx  # bbox的id
                # cv2.imwrite(f"./temp/{info_annotation['id']}.jpg", img_copy)
                info_annotation["segmentation"] = [
                    [xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax]
                ]  # 四个点的坐标
                info_annotation["iscrowd"] = 0  # 单例
                annotations.append(info_annotation)
    return images, annotations


def gen_json_file(yolov8_data_path, coco_format_path, key):
    # json path
    json_path = os.path.join(coco_format_path, f"annotations/instances_{key}.json")
    dst_path = os.path.join(coco_format_path, f"{key}")

    if not os.path.exists(os.path.dirname(json_path)):
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
    data_path = os.path.join(yolov8_data_path, f"{key}/images")
    label_path = os.path.join(yolov8_data_path, f"{key}/labels")
    images, anns = gen1_yolo_covert_coco_format(data_path, label_path)
    if key == "train":
        train_data["images"] = images
        train_data["annotations"] = anns
        with open(json_path, "w") as f:
            json.dump(train_data, f, indent=2)
        # shutil.copy(data_path,'')
    elif key == "test":
        test_data["images"] = images
        test_data["annotations"] = anns
        with open(json_path, "w") as f:
            json.dump(test_data, f, indent=2)
    elif key == "val":
        valid_data["images"] = images
        valid_data["annotations"] = anns
        with open(json_path, "w") as f:
            json.dump(valid_data, f, indent=2)
    else:
        print(f"key is {key}")
    print(f"generate {key} json success!")
    return


if __name__ == "__main__":

    yolov8_data_path = "/data/hzh/Gen1"
    coco_format_path = "/data/hzh/Gen1"
    gen_json_file(yolov8_data_path, coco_format_path, key="train")
    gen_json_file(yolov8_data_path, coco_format_path, key="test")
    gen_json_file(yolov8_data_path, coco_format_path, key="val")
    # gen_json_file(yolov8_data_path, coco_format_path,key='val')
