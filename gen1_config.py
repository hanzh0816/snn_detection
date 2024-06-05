img_scale = (640, 640)  # width, height

# dataset settings
data_root = "data/Gen1/"
dataset_type = "CocoDataset"
classes = ("cars", "pedestrians")

backend_args = None


val_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    metainfo=dict(classes=classes),
    ann_file="annotations/instances_val.json",
    data_prefix=dict(img="val/images"),
    pipeline=[
        dict(type="LoadImageFromNDarrayFile", backend_args=backend_args),
        dict(type="LoadAnnotations", with_bbox=True),
        dict(type="PackDetInputs"),
    ],
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    backend_args=backend_args,
)


val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=val_dataset,
)

vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(
    type="DetLocalVisualizer", vis_backends=vis_backends, name="visualizer"
)
log_processor = dict(type="LogProcessor", window_size=50, by_epoch=True)
