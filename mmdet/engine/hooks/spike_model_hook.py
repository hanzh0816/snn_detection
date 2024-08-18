from mmengine.hooks import Hook
from mmdet.registry import HOOKS
from spikingjelly.clock_driven import functional


@HOOKS.register_module()
class SpikeResetHook(Hook):

    def __init__(self):
        pass

    # def before_train_iter(self, runner, *kargs, **kwargs) -> None:
    #     functional.reset_net(runner.model)

    def after_train_iter(self, runner, *kargs, **kwargs) -> None:
        functional.reset_net(runner.model)

    def after_val_iter(self, runner, *kargs, **kwargs) -> None:
        functional.reset_net(runner.model)

    def after_test_iter(self, runner, *kargs, **kwargs) -> None:
        functional.reset_net(runner.model)
