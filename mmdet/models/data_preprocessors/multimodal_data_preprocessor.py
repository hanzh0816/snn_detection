from mmdet.registry import MODELS
from mmdet.structures import TrackDataSample
from mmdet.structures.mask import BitmapMasks
from .data_preprocessor import DetDataPreprocessor
from mmdet.models.utils.misc import samplelist_boxtype2tensor

from mmengine.model.utils import stack_batch


@MODELS.register_module()
class MultiModalDetDataPreprocessor(DetDataPreprocessor):
    def forward(self, data: dict, training: bool = False) -> dict:
        batch_pad_shape = self._get_pad_shape(data)
        # 转换数据到对应device
        data = self.cast_data(data)
        inputs, event, data_samples = data["inputs"], data["event"], data["data_samples"]
        batch_inputs = []
        batch_events = []
        for _batch_input, _batch_event in zip(inputs, event):
            _batch_input = _batch_input.float()
            _batch_event = _batch_event.float()
            batch_inputs.append(_batch_input)
            batch_events.append(_batch_event)

        inputs = stack_batch(batch_inputs)
        event = stack_batch(batch_events)

        # snn_todo 缺少image的归一化操作

        if data_samples is not None:
            # NOTE the batched image size information may be useful, e.g.
            # in DETR, this is needed for the construction of masks, which is
            # then used for the transformer_head.
            batch_input_shape = tuple(inputs[0].size()[-2:])
            for data_sample, pad_shape in zip(data_samples, batch_pad_shape):
                data_sample.set_metainfo(
                    {"batch_input_shape": batch_input_shape, "pad_shape": pad_shape}
                )
            if self.boxtype2tensor:
                samplelist_boxtype2tensor(data_samples)

        #     if self.pad_mask and training:
        #         self.pad_gt_masks(data_samples)

        #     if self.pad_seg and training:
        #         self.pad_gt_sem_seg(data_samples)

        # if training and self.batch_augments is not None:
        #     for batch_aug in self.batch_augments:
        #         inputs, data_samples = batch_aug(inputs, data_samples)

        return {"inputs": inputs, "event": event, "data_samples": data_samples}
