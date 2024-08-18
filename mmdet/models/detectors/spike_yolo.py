from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .single_stage import SingleStageDetector


@MODELS.register_module()
class SpikeYOLO(SingleStageDetector):
    r"""Implementation of SpikeYOLO in 2024
    <https://arxiv.org/abs/2407.20708v1>`_

    Args:
        backbone (dict): Config dict for the backbone.
        bbox_head (dict): Config dict for the bbox head.
        train_cfg (dict, optional): Training config dict. Defaults to None.
        test_cfg (dict, optional): Testing config dict. Defaults to None.
        data_preprocessor (dict, optional): Config dict for the data preprocessor.
        init_cfg (dict, optional): Initialization config dict. Defaults to None.
    """

    def __init__(
        self,
        backbone: ConfigType,
        neck: ConfigType,
        bbox_head: ConfigType,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        data_preprocessor: OptConfigType = None,
        init_cfg: OptMultiConfig = None,
    ) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
        )
