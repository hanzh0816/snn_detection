_base_ = "./spike_yolo_s_2x12_100e_coco.py"

scale = "m"
model = dict(
    backbone=dict(type="SpikeYOLOBackbone", scale=scale),
    neck=dict(type="SpikeYOLONeck", scale=scale),
    bbox_head=dict(
        type="SpikeYOLOHead",
        scale=scale,
    ),
)


batch_size = 5

train_dataloader = dict(batch_size=batch_size)
val_dataloader = dict(batch_size=1)
