data_root = "/data2/hzh/DSEC-COCO/"
backend_args = None
metainfo = dict(
    classes=("pedestrian", "rider", "car", "bus", "truck", "bicycle", "motorcycle", "train"),
    pallete=[
        (220, 20, 60),
        (119, 11, 32),
        (0, 0, 142),
        (0, 0, 230),
        (106, 0, 228),
        (0, 60, 100),
        (0, 80, 100),
        (0, 0, 70),
    ],
)

train_pipeline = [
    dict(type="LoadImageAndEventFromFolder"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="PackMultiModalDetInputs"),
]

test_pipeline = [
    dict(type="LoadImageAndEventFromFolder"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(type="PackMultiModalDetInputs"),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=dict(type="AspectRatioBatchSampler"),
    dataset=dict(
        type="CocoDataset",
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img="train/"),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        ann_file="annotations/train.json",
        pipeline=train_pipeline,
        backend_args=backend_args,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="CocoDataset",
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img="test/"),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        ann_file="annotations/test.json",
        pipeline=train_pipeline,
        backend_args=backend_args,
    ),
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type="CocoMetric",
    ann_file=data_root + "annotations/test.json",
    metric="bbox",
    format_only=False,
    backend_args=backend_args,
)
test_evaluator = val_evaluator
