import torch

from .hook import Hook


class ParamUpdateHook(Hook):
    def __init__(self) -> None:
        super().__init__()
    
    # specific param_update function, called inside train_step of each algorithm
    # def param_update(self, algorithm, loss):
    #     # algorithm.optimizer.zero_grad()
    #     # update parameters
    #     if algorithm.use_amp:
    #         algorithm.loss_scaler.scale(loss).backward()
    #         if (algorithm.clip_grad > 0):
    #             algorithm.loss_scaler.unscale_(algorithm.optimizer)
    #             torch.nn.utils.clip_grad_norm_(algorithm.model.parameters(), algorithm.clip_grad)
    #         algorithm.loss_scaler.step(algorithm.optimizer)
    #         algorithm.loss_scaler.update()
    #     else:
    #         loss.backward()
    #         if (algorithm.clip_grad > 0):
    #             torch.nn.utils.clip_grad_norm_(algorithm.model.parameters(), algorithm.clip_grad)
    #         algorithm.optimizer.step()

    #     algorithm.scheduler.step()
    #     algorithm.model.zero_grad()
    def param_update(self, algorithm, loss):
        # algorithm.optimizer.zero_grad()
        # update parameters
        if algorithm.use_amp:
            algorithm.loss_scaler_1.scale(loss['model_1']).backward()
            algorithm.loss_scaler_2.scale(loss['model_2']).backward()
            if (algorithm.clip_grad > 0):
                algorithm.loss_scaler_1.unscale_(algorithm.optimizer['model_1'])
                algorithm.loss_scaler_2.unscale_(algorithm.optimizer['model_2'])
                torch.nn.utils.clip_grad_norm_(algorithm.model_1.parameters(), algorithm.clip_grad)
                torch.nn.utils.clip_grad_norm_(algorithm.model_2.parameters(), algorithm.clip_grad)
            algorithm.loss_scaler_1.step(algorithm.optimizer['model_1'])
            algorithm.loss_scaler_2.step(algorithm.optimizer['model_2'])
            algorithm.loss_scaler_1.update()
            algorithm.loss_scaler_2.update()
        else:
            loss['model_1'].backward(retain_graph=True)
            loss['model_2'].backward()
            if (algorithm.clip_grad > 0):
                torch.nn.utils.clip_grad_norm_(algorithm.model_1.parameters(), algorithm.clip_grad)
                torch.nn.utils.clip_grad_norm_(algorithm.model_2.parameters(), algorithm.clip_grad) 
    
            algorithm.optimizer['model_1'].step()
            algorithm.optimizer['model_2'].step()
        algorithm.scheduler['model_1'].step()
        algorithm.scheduler['model_2'].step()
        algorithm.model_1.zero_grad()
        algorithm.model_2.zero_grad()
    