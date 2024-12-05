dataset_type = "CocoDataset"
data_root = "/data2/hzh/DSEC-COCO-balanced-mini/"
backend_args = None

metainfo = dict(
    classes=("pedestrian", "car"),
    pallete=[
        (220, 20, 60),
        (119, 11, 32),
    ],
)

# train dataloader settings
train_pipeline = [
    dict(type="LoadImageAndEventFromFolder"),
    dict(type="LoadAnnotations", with_bbox=True),
    # dict(type="Resize", scale=(1333, 800), keep_ratio=True),
    dict(type="PackMultiModalDetInputs"),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    batch_sampler=dict(type="AspectRatioBatchSampler", drop_last=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img="train/"),
        ann_file="annotations/train.json",
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args,
    ),
)


test_pipeline = [
    dict(type="LoadImageAndEventFromFolder"),
    dict(type="LoadAnnotations", with_bbox=True),
    # dict(type="Resize", scale=(1333, 800), keep_ratio=True),
    dict(type="PackMultiModalDetInputs"),
]

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        data_prefix=dict(img="test/"),
        ann_file="annotations/test.json",
        test_mode=True,
        pipeline=test_pipeline,
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
