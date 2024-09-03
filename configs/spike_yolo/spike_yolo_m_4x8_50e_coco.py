_base_ = [
    "spike_yolo_m_baseline_2x8_100e_coco.py",
]

batch_size = 8
train_dataloader = dict(batch_size=batch_size)
val_dataloader = dict(batch_size=1)

base_lr = 1e-3
max_epochs = 50
num_last_epochs = 15
end_warmup_epoch = 8
interval = 5

train_cfg = dict(max_epochs=max_epochs)

optim_wrapper = dict(optimizer=dict(lr=base_lr))
param_scheduler = [
    dict(
        # use quadratic formula to warm up epochs
        # and lr is updated by iteration
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

# vis_backends = [
#     dict(type="LocalVisBackend"),
#     dict(
#         type="WandbVisBackend",
#         init_kwargs=dict(project="snn_detection", name="spike_yolo_m_4x8_50e_coco"),
#     ),
# ]
# visualizer = dict(vis_backends=vis_backends)
