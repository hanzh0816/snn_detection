_base_ = [
    "../_base_/default_runtime.py",
]

# ================================================================================================
# training settings
max_epochs = 100
num_last_epochs = 15
end_warmup_epoch = 2
interval = 5

# runtime settings
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs, val_interval=interval)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

# optimizer
# default lr: 8 gpus x 2 sample = 16
base_lr = 0.01
auto_scale_lr = dict(enable=True, base_batch_size=16)
weight_decay = 5e-4
momentum = 0.9

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(
        type="SGD", lr=base_lr, momentum=momentum, weight_decay=weight_decay, nesterov=True
    ),
)
# learning rate scheduler
param_scheduler = [
    dict(
        # use quadratic formula to warm up epochs
        # and lr is updated by iteration
        # TODO: fix default scope in get function
        type="mmdet.QuadraticWarmupLR",
        by_epoch=True,
        begin=0,
        end=end_warmup_epoch,
        convert_to_iter_based=True,
    ),
    dict(
        # use cosine lr from warmup end epoch to last epochs
        type="CosineAnnealingLR",
        eta_min=base_lr * 0.05,
        begin=end_warmup_epoch,
        T_max=max_epochs - num_last_epochs,
        end=max_epochs - num_last_epochs,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
    dict(
        # use fixed lr during last epochs
        type="ConstantLR",
        by_epoch=True,
        factor=1,
        begin=max_epochs - num_last_epochs,
        end=max_epochs,
    ),
]

# ================================================================================================
# dataset settings

data_root = "data/coco/"
dataset_type = "CocoDataset"
batch_size = 12
img_scale = (640, 640)
backend_args = None

# train data settings
pre_transform = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="Resize", scale=img_scale, keep_ratio=True),
    dict(type="Pad", pad_to_square=True, pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type="LoadAnnotations", with_bbox=True),
]

last_transform = [
    dict(
        type="mmdet.PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]
train_pipeline = [
    *pre_transform,
    *last_transform,
]

train_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    ann_file="annotations/instances_train2017.json",
    data_prefix=dict(img="train2017/"),
    pipeline=train_pipeline,
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    backend_args=backend_args,
)
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=train_dataset,
)

# test & val data settings
test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="Resize", scale=img_scale, keep_ratio=True),
    dict(type="Pad", pad_to_square=True, pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]
val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="annotations/instances_val2017.json",
        data_prefix=dict(img="val2017/"),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type="CocoMetric",
    ann_file=data_root + "annotations/instances_val2017.json",
    metric="bbox",
    backend_args=backend_args,
)
test_evaluator = val_evaluator

# ================================================================================================
# model settings

scale = "s"

model_train_cfg = dict(
    assigner=dict(
        type="BatchTaskAlignedAssigner",
        num_classes=80,
        use_ciou=True,
        topk=10,
        alpha=0.5,
        beta=6.0,
        eps=1e-9,
    )
)

model_test_cfg = dict(
    # The config of multi-label for multi-class prediction.
    multi_label=True,
    # The number of boxes before NMS
    nms_pre=30000,
    score_thr=0.01,  # Threshold to filter out boxes.
    nms=dict(type="nms", iou_threshold=0.65),  # NMS type and threshold
    max_per_img=300,
)  # Max number of detections of each image

init_cfg = dict(type="Kaiming", layer=["Conv2d", "Linear"], distribution="uniform")

model = dict(
    type="SpikeYOLO",
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
    ),
    backbone=dict(type="SpikeYOLOBackbone", scale=scale),
    neck=dict(type="SpikeYOLONeck", scale=scale),
    bbox_head=dict(
        type="SpikeYOLOHead",
        scale=scale,
        loss_cls=dict(
            type="CrossEntropyLoss",
            use_sigmoid=True,
            reduction="none",
            loss_weight=0.5,
        ),
        loss_bbox=dict(
            type="YOLOv8IoULoss",
            iou_mode="ciou",
            bbox_format="xyxy",
            reduction="sum",
            loss_weight=7.5,
            return_iou=False,
        ),
        loss_dfl=dict(type="DistributionFocalLoss", reduction="mean", loss_weight=1.5 / 4),
    ),
    train_cfg=model_train_cfg,
    test_cfg=model_test_cfg,
    init_cfg=init_cfg,
)

# ================================================================================================
# hook settings

custom_hooks = [
    dict(type="SpikeResetHook"),
]
default_hooks = dict(checkpoint=dict(type="CheckpointHook", max_keep_ckpts=3, save_best="auto"))


# ================================================================================================
# other settings

randomness = dict(seed=41, diff_rank_seed=True)
