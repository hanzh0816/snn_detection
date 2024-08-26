_base_ = "./spike_yolo_m_t4_cupy_2x5_100e_coco.py"

batch_size = 12

train_dataloader = dict(batch_size=batch_size)