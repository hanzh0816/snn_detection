_base_ = "./spike_yolo_m_2x5_100e_coco.py"
scale = "m"
model = dict(
    backbone=dict(type="SpikeYOLOBackbone", scale=scale, T=4, lif_backend="cupy"),
    neck=dict(type="SpikeYOLONeck", scale=scale, lif_backend="cupy"),
    bbox_head=dict(type="SpikeYOLOHead", scale=scale),
)
