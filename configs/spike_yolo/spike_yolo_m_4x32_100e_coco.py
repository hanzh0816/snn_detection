_base_ = [
    "spike_yolo_m_baseline_2x8_100e_coco.py",
]

batch_size = 32
train_dataloader = dict(batch_size=batch_size)
val_dataloader = dict(batch_size=16)


vis_backends = [
    dict(type="LocalVisBackend"),
    dict(
        type="WandbVisBackend",
        init_kwargs=dict(project="snn_detection", name="spike_yolo_m_4x32_100e_coco"),
    ),
]
visualizer = dict(vis_backends=vis_backends)
