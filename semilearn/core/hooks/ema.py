from .hook import Hook
from semilearn.core.utils import EMA


class EMAHook(Hook):
    def __init__(self) -> None:
        super().__init__()
    
    def before_run(self, algorithm):
        # algorithm.ema = EMA(algorithm.model, algorithm.ema_m)
        # algorithm.ema.register()
        # if algorithm.resume:
        #     algorithm.ema.load(algorithm.ema_model)
        algorithm.ema_1 = EMA(algorithm.model_1, algorithm.ema_m)
        algorithm.ema_2 = EMA(algorithm.model_2, algorithm.ema_m)
        algorithm.ema_1.register()
        algorithm.ema_2.register()
        if algorithm.resume:
            algorithm.ema_1.load(algorithm.ema_model_1)
            algorithm.ema_2.load(algorithm.ema_model_2)

    def after_train_step(self, algorithm):
        # if algorithm.ema is not None:
        #     algorithm.ema.update()
        #     algorithm.ema_model.load_state_dict(algorithm.model.state_dict())
        #     algorithm.ema_model.load_state_dict(algorithm.ema.shadow, strict=False)
        if algorithm.ema_1 is not None:
            algorithm.ema_1.update()
            algorithm.ema_model_1.load_state_dict(algorithm.model_1.state_dict())
            algorithm.ema_model_1.load_state_dict(algorithm.ema_1.shadow, strict=False)
        if algorithm.ema_2 is not None:
            algorithm.ema_2.update()
            algorithm.ema_model_2.load_state_dict(algorithm.model_2.state_dict())
            algorithm.ema_model_2.load_state_dict(algorithm.ema_2.shadow, strict=False)
