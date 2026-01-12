import os
import contextlib
import numpy as np
from inspect import signature
from collections import OrderedDict
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, \
    confusion_matrix, roc_auc_score, roc_curve, auc

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from semilearn.core.hooks import Hook, get_priority, CheckpointHook, TimerHook, LoggingHook, DistSamplerSeedHook, \
    ParamUpdateHook, EvaluationHook, EMAHook
from semilearn.core.utils import get_dataset, get_data_loader, get_optimizer, get_cosine_schedule_with_warmup, \
    Bn_Controller

from PIL import Image
from PIL import Image, ImageDraw
import torch.nn as nn

output_folder = '' #For anonymous review, we removed all paths in the code to prevent unintended disclosure of author identity.

class PadPrompter(nn.Module):
    def __init__(self, args):
        super(PadPrompter, self).__init__()
        self.base_size = 16
        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])
        self.dataset_name = args.dataset
    def forward(self, x, token_indices, topk):
        B, C, H, W = x.shape
        for i in range(x.shape[0]):
            # 将张量转换为图片
            image = (x[i].cpu() * self.std.view(-1, 1, 1)) + self.mean.view(-1, 1, 1)
            image = torch.clamp(image, 0, 1)
            image = image * 255
            image_np = image.detach().permute(1, 2, 0).numpy().astype(np.uint8)
            image = Image.fromarray(image_np)

            # 绘制红色边框
            draw = ImageDraw.Draw(image)
            for j in range(topk):
                y0, x0 = divmod(token_indices[i][j].item(), W // self.base_size)
                y0, x0 = y0 * self.base_size, x0 * self.base_size
                draw.rectangle([x0, y0, x0 + self.base_size, y0 + self.base_size], outline="red", width=2)
            
            # 保存图片
            image_filename = f"image_{self.dataset_name}_{i}.jpg"
            image_path = f"{output_folder}/{image_filename}"
            image.save(image_path)
        
        return x

def tsne_visualization(feats, labels, dataset_name, save_path, x_min=None, x_max=None):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import os
    
    num_classes = 20  # 改成20个类
    feats = torch.cat(feats, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    
    # 只保留前20个类别的标签
    valid_labels = np.unique(labels)[num_classes:]
    mask = np.isin(labels, valid_labels)
    feats = feats[mask]
    labels = labels[mask]

    tsne = TSNE(n_components=2, random_state=0)
    feats_tsne = tsne.fit_transform(feats)

    plt.figure(figsize=(10, 8))
    cmap = plt.cm.get_cmap('tab20', num_classes)  # 使用离散的颜色映射
    scatter = plt.scatter(feats_tsne[:, 0], feats_tsne[:, 1], c=labels, cmap=cmap, alpha=0.7)
    plt.colorbar(scatter, ticks=np.arange(num_classes))

    plt.title("T-SNE Visualization of Features")
    plt.xlabel("T-SNE Dimension 1")
    plt.ylabel("T-SNE Dimension 2")
    if x_min is not None and x_max is not None:
        plt.xlim(x_min, x_max)
    save_path = os.path.join(save_path, f"{dataset_name}.png")
    plt.savefig(save_path)
    plt.close()
    
import sklearn.metrics as sk
def get_measures(_pos, _neg, recall_level=0.95):
      pos = np.array(_pos[:]).reshape((-1, 1))
      neg = np.array(_neg[:]).reshape((-1, 1))
      examples = np.squeeze(np.vstack((pos, neg)))
      labels = np.zeros(len(examples), dtype=np.int32)
      labels[:len(pos)] += 1
  
      auroc = sk.roc_auc_score(labels, examples)
  
      return auroc

class AlgorithmBase:
    """
        Base class for algorithms
        init algorithm specific parameters and common parameters
        
        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
    """

    def __init__(
            self,
            args,
            net_builder,
            tb_log=None,
            logger=None,
            **kwargs):

        # common arguments
        self.tb_dict = None
        self.args = args
        self.num_classes = args.num_classes
        self.ema_m = args.ema_m
        self.epochs = args.epoch
        self.num_train_iter = args.num_train_iter
        self.num_eval_iter = args.num_eval_iter
        self.num_log_iter = args.num_log_iter
        self.num_iter_per_epoch = int(self.num_train_iter // self.epochs)
        self.lambda_u = args.ulb_loss_ratio
        self.use_cat = args.use_cat
        self.use_amp = args.use_amp
        self.clip_grad = args.clip_grad
        self.save_name = args.save_name
        self.save_dir = args.save_dir
        self.resume = args.resume
        self.algorithm = args.algorithm

        # commaon utils arguments
        self.tb_log = tb_log
        self.print_fn = print if logger is None else logger.info
        self.ngpus_per_node = torch.cuda.device_count()
        self.loss_scaler_1 = GradScaler()
        self.loss_scaler_2 = GradScaler()
        self.amp_cm = autocast if self.use_amp else contextlib.nullcontext
        self.gpu = args.gpu
        self.rank = args.rank
        self.distributed = args.distributed
        self.world_size = args.world_size

        # common model related parameters
        self.epoch = 0
        self.it = 0
        self.best_eval_acc, self.best_it = 0.0, 0
        self.bn_controller = Bn_Controller()
        self.net_builder = net_builder
        # self.ema = None
        self.ema_1 = None
        self.ema_2 = None

        # build dataset
        self.dataset_dict = self.set_dataset()

        # build data loader
        self.loader_dict = self.set_data_loader()

        # cv, nlp, speech builder different arguments
        # self.model = self.set_model()
        # self.ema_model = self.set_ema_model()
        self.model_1 = self.set_model()['model_1']
        self.model_2 = self.set_model()['model_2']
        self.ema_model_1 = self.set_ema_model()['ema_model_1']
        self.ema_model_2 = self.set_ema_model()['ema_model_2']

        # build optimizer and scheduler
        self.optimizer, self.scheduler = self.set_optimizer()

        # other arguments specific to the algorithm
        # self.init(**kwargs)

        # set common hooks during training
        self._hooks = []  # record underlying hooks 
        self.hooks_dict = OrderedDict()  # actual object to be used to call hooks
        self.set_hooks()

    def init(self, **kwargs):
        """
        algorithm specific init function, to add parameters into class
        """
        raise NotImplementedError

    def set_dataset(self):
        if self.rank != 0 and self.distributed:
            torch.distributed.barrier()
        dataset_dict = get_dataset(self.args, self.algorithm, self.args.dataset, self.args.num_labels,
                                   self.args.num_classes, self.args.data_dir)
        self.args.ulb_dest_len = len(dataset_dict['train_ulb']) if dataset_dict['train_ulb'] is not None else 0
        self.args.lb_dest_len = len(dataset_dict['train_lb'])
        self.print_fn(
            "unlabeled data number: {}, labeled data number {}".format(self.args.ulb_dest_len, self.args.lb_dest_len))
        if self.rank == 0 and self.distributed:
            torch.distributed.barrier()
        return dataset_dict

    def set_data_loader(self):
        self.print_fn("Create train and test data loaders")
        loader_dict = {}
        loader_dict['train_lb'] = get_data_loader(self.args,
                                                  self.dataset_dict['train_lb'],
                                                  self.args.batch_size,
                                                  data_sampler=self.args.train_sampler,
                                                  num_iters=self.num_train_iter,
                                                  num_epochs=self.epochs,
                                                  num_workers=self.args.num_workers,
                                                  distributed=self.distributed)

        loader_dict['train_ulb'] = get_data_loader(self.args,
                                                   self.dataset_dict['train_ulb'],
                                                   self.args.batch_size * self.args.uratio,
                                                   data_sampler=self.args.train_sampler,
                                                   num_iters=self.num_train_iter,
                                                   num_epochs=self.epochs,
                                                   num_workers=2 * self.args.num_workers,
                                                   distributed=self.distributed)

        loader_dict['id_dset'] = get_data_loader(self.args,
                                              self.dataset_dict['id_dset'],
                                              self.args.eval_batch_size,
                                              # make sure data_sampler is None for evaluation
                                              data_sampler=None,
                                              num_workers=self.args.num_workers,
                                              drop_last=False)
        loader_dict['ood_dset'] = get_data_loader(self.args,
                                              self.dataset_dict['ood_dset'],
                                              self.args.eval_batch_size,
                                              # make sure data_sampler is None for evaluation
                                              data_sampler=None,
                                              num_workers=self.args.num_workers,
                                              drop_last=False)

        loader_dict['eval'] = get_data_loader(self.args,
                                              self.dataset_dict['eval'],
                                              self.args.eval_batch_size,
                                              # make sure data_sampler is None for evaluation
                                              data_sampler=None,
                                              num_workers=self.args.num_workers,
                                              drop_last=False)

        if self.dataset_dict['test'] is not None:
            if isinstance(self.dataset_dict['test'], dict):
                loader_dict['test'] = {}
                for k, v in self.dataset_dict['test'].items():
                    if v is not None:
                        loader_dict['test'][k] = get_data_loader(self.args,
                                                                 self.dataset_dict['test'][k],
                                                                 self.args.eval_batch_size,
                                                                 # make sure data_sampler is None for evaluation
                                                                 data_sampler=None,
                                                                 num_workers=self.args.num_workers,
                                                                 drop_last=False)
            else:
                loader_dict['test'] = get_data_loader(self.args,
                                                      self.dataset_dict['test'],
                                                      self.args.eval_batch_size,
                                                      # make sure data_sampler is None for evaluation
                                                      data_sampler=None,
                                                      num_workers=self.args.num_workers,
                                                      drop_last=False)
        self.print_fn(f'[!] data loader keys: {loader_dict.keys()}')
        return loader_dict

    # def set_optimizer(self):
    #     self.print_fn("Create optimizer and scheduler")
    #     optimizer = get_optimizer(self.model, self.args.optim, self.args.lr, self.args.momentum, self.args.weight_decay,
    #                               self.args.layer_decay)
    #     scheduler = get_cosine_schedule_with_warmup(optimizer,
    #                                                 self.num_train_iter,
    #                                                 num_warmup_steps=self.args.num_warmup_iter)
    #     return optimizer, scheduler
    def set_optimizer(self):
        self.print_fn("Create optimizer and scheduler")
        optimizer_dict = {
            # 'v_prompt_learner': get_optimizer(self.v_prompt_learner, self.args.optim, self.args.lr, self.args.momentum, self.args.weight_decay,
            #                                    self.args.layer_decay),
            'model_1': get_optimizer(self.model_1, self.args.optim, self.args.lr, self.args.momentum, self.args.weight_decay,
                                  self.args.layer_decay),
            'model_2': get_optimizer(self.model_2, self.args.optim, self.args.lr, self.args.momentum, self.args.weight_decay,
                                  self.args.layer_decay)
            }
        scheduler_dict = {
            'model_1': get_cosine_schedule_with_warmup(optimizer_dict['model_1'],
                                                    self.num_train_iter,
                                                    num_warmup_steps=self.args.num_warmup_iter),
            'model_2': get_cosine_schedule_with_warmup(optimizer_dict['model_2'],
                                                     self.num_train_iter,
                                                     num_warmup_steps=self.args.num_warmup_iter)
            }

        return optimizer_dict, scheduler_dict

    def set_model(self):
        model = self.net_builder(num_classes=self.num_classes, pretrained=self.args.use_pretrain,
                                 pretrained_path=self.args.pretrain_path)
        return model

    def set_ema_model(self):
        """
        initialize ema model from model
        """
        ema_model = self.net_builder(num_classes=self.num_classes)
        ema_model.load_state_dict(self.model.state_dict())
        return ema_model

    def set_hooks(self):
        """
        register necessary training hooks
        """
        # parameter update hook is called inside each train_step
        self.register_hook(TimerHook(), None, "HIGHEST")
        self.register_hook(EMAHook(), None, "HIGHEST")
        self.register_hook(EvaluationHook(), None, "HIGHEST")
        self.register_hook(CheckpointHook(), None, "VERY_HIGH")
        self.register_hook(DistSamplerSeedHook(), None, "HIGH")
        self.register_hook(LoggingHook(), None, "LOW")

        # for hooks to be called in train_step, name it for simpler calling
        self.register_hook(ParamUpdateHook(), "ParamUpdateHook")

    def process_batch(self, **kwargs):
        """
        process batch data, send data to cuda
        NOTE **kwargs should have the same arguments to train_step function as keys to work properly
        """
        input_args = signature(self.train_step).parameters
        input_args = list(input_args.keys())
        input_dict = {}

        for arg, var in kwargs.items():
            if not arg in input_args:
                continue

            if var is None:
                continue

            # send var to cuda
            if isinstance(var, dict):
                var = {k: v.cuda(self.gpu) for k, v in var.items()}
            else:
                var = var.cuda(self.gpu)
            input_dict[arg] = var
        return input_dict

    def train_step(self, *args, **kwargs):
        """
        train_step specific to each algorithm
        """
        # implement train step for each algorithm
        # compute loss
        # update model 
        # record tb_dict
        # return tb_dict
        raise NotImplementedError

    def train(self):
        """
        train function
        """
        # self.model.train()
        self.model_1.train()
        self.model_2.train()
        self.call_hook("before_run")

        for epoch in range(self.epoch, self.epochs):
            self.epoch = epoch

            # prevent the training iterations exceed args.num_train_iter
            if self.it >= self.num_train_iter:
                break

            self.call_hook("before_train_epoch")

            for data_lb, data_ulb in zip(self.loader_dict['train_lb'],
                                         self.loader_dict['train_ulb']):
                # prevent the training iterations exceed args.num_train_iter
                if self.it >= self.num_train_iter:
                    break

                self.call_hook("before_train_step")
                self.tb_dict = self.train_step(**self.process_batch(**data_lb, **data_ulb))
                self.call_hook("after_train_step")
                self.it += 1

            self.call_hook("after_train_epoch")

        self.call_hook("after_run")
    
      
    def evaluate(self, eval_dest='eval', return_logits=False):
        """
        evaluation function
        """
        # self.model.eval()
        # self.ema.apply_shadow()
        # self.v_prompt_learner.eval()
        self.model_1.eval()
        self.model_2.eval()
        self.ema_1.apply_shadow()
        self.ema_2.apply_shadow()
        eval_loader = self.loader_dict[eval_dest]
        # for OOD dectection
        id_loader = self.loader_dict['id_dset']
        ood_loader = self.loader_dict['ood_dset']

        total_loss = 0.0
        total_num = 0.0
        y_true = []
        y_pred = []
        # y_probs = []
        y_logits = []
        visual_features = []
        visual_labels = []

        id_logits = []
        id_labels = []  # ID标签（ID = 1）
        ood_logits = []
        ood_labels = []  # OOD标签（OOD = 0）
        to_np = lambda x: x.data.cpu().numpy()
        concat = lambda x: np.concatenate(x, axis=0)
        id_scores, ood_scores = [], []
        aurocs = []

        with torch.no_grad():
            for data in id_loader:
                x = data['x_lb']
                y = data['y_lb']

                if isinstance(x, dict):
                    x = {k: v.cuda(self.gpu) for k, v in x.items()}
                else:
                    x = x.cuda(self.gpu)
                y = y.cuda(self.gpu)
                
                # Model 1: 原始文本
                outputs_1 = self.model_1(x, None, labels=None, use_enhanced_text=False, update_cache=False)
                # Model 2: 增强文本
                outputs_2 = self.model_2(x, None, labels=None, use_enhanced_text=True, update_cache=False)
                
                logits_1 = outputs_1['logits']
                logits_2 = outputs_2['logits']
                logits = logits_2
                smax = to_np(F.softmax(logits / 1, dim=-1))
                id_scores.append(-np.max(smax, axis=1))
                

            for data in ood_loader:
                x = data['x_lb']
                y = data['y_lb']

                if isinstance(x, dict):
                    x = {k: v.cuda(self.gpu) for k, v in x.items()}
                else:
                    x = x.cuda(self.gpu)
                y = y.cuda(self.gpu)
                
                # Model 1: 原始文本
                outputs_1 = self.model_1(x, None, labels=None, use_enhanced_text=False, update_cache=False)
                # Model 2: 增强文本
                outputs_2 = self.model_2(x, None, labels=None, use_enhanced_text=True, update_cache=False)
                
                logits_1 = outputs_1['logits']
                logits_2 = outputs_2['logits']
                logits = (logits_1 + logits_2) / 2
                smax = to_np(F.softmax(logits / 1, dim=-1))
                ood_scores.append(-np.max(smax, axis=1))
        in_score = concat(id_scores)
        out_score = concat(ood_scores)
        measures = get_measures(-in_score, -out_score)
        aurocs.append(measures)
        auroc = np.mean(aurocs)
        print('auroc', auroc)
       

        with torch.no_grad():
            for data in eval_loader:
                x = data['x_lb']
                y = data['y_lb']
                
                if isinstance(x, dict):
                    x = {k: v.cuda(self.gpu) for k, v in x.items()}
                else:
                    x = x.cuda(self.gpu)
                y = y.cuda(self.gpu)

                num_batch = y.shape[0]
                total_num += num_batch
                
                # Model 1: 使用原始文本
                outputs_1 = self.model_1(
                    x, 
                    None,
                    labels=None,  # 评估时不需要labels
                    use_enhanced_text=False,  # 使用原始文本
                    update_cache=False  # 不更新cache
                )
                
                # Model 2: 使用增强文本
                outputs_2 = self.model_2(
                    x, 
                    None,
                    labels=None,
                    use_enhanced_text=True,  # 使用增强文本
                    update_cache=False
                )
                
                logits_1 = outputs_1['logits']
                logits_2 = outputs_2['logits']
                token_index = outputs_1['token_index']
                w = PadPrompter(self.args)(x, token_index, self.args.topk_patches)
                
                logits = (logits_1 + logits_2) / 2
                # ### t-sne visualization
                # visual_features.append(logits.cpu())
                # visual_labels.append(y.cpu())
                # ### t-sne visualization
                # loss = F.cross_entropy(logits, y, reduction='mean')
                y_true.extend(y.cpu().tolist())
                y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
                y_logits.append(logits.cpu().numpy())
                # y_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
                # total_loss += loss.item() * num_batch
        
        y_true = np.array(y_true)

        # closed_mask = y_true < self.num_classes
        # open_mask = y_true >= self.num_classes
        # y_true[open_mask] = self.num_classes
        
        # closed_mask = y_true >= 0
        # open_mask = y_true < 0
        closed_mask = y_true < self.num_classes
        open_mask = y_true >= self.num_classes
        y_true[open_mask] = self.num_classes
        y_pred = np.array(y_pred)
        y_logits = np.concatenate(y_logits)
        print('closed', len(y_true[closed_mask]))
        print('test_num', len(y_pred))

        top1 = accuracy_score(y_true[closed_mask], y_pred[closed_mask])
        balanced_top1 = balanced_accuracy_score(y_true, y_pred)
        # top1 = accuracy_score(y_true, y_pred)
        # balanced_top1 = balanced_accuracy_score(y_true, y_pred)
        # precision = precision_score(y_true, y_pred, average='macro')
        # recall = recall_score(y_true, y_pred, average='macro')
        # F1 = f1_score(y_true, y_pred, average='macro')
        print("top1:", top1)
        print('balanced_top1', balanced_top1)
        if self.num_classes <= 10:
            cf_mat = confusion_matrix(y_true, y_pred, normalize='true')
            self.print_fn('confusion matrix:\n' + np.array_str(cf_mat))
        # self.ema.restore()
        # self.model.train()
        self.ema_1.restore()
        self.ema_2.restore()
        self.model_1.train()
        self.model_2.train()

        # eval_dict = {eval_dest + '/loss': total_loss / total_num, eval_dest + '/top-1-acc': top1,
        #              eval_dest + '/balanced_acc': balanced_top1, eval_dest + '/precision': precision,
        #              eval_dest + '/recall': recall, eval_dest + '/F1': F1}

        eval_dict = {eval_dest + '/loss': total_loss / total_num, eval_dest + '/top-1-acc': top1, eval_dest + '/balanced_acc': balanced_top1}

        if return_logits:
            eval_dict[eval_dest + '/logits'] = y_logits
        return eval_dict

    def get_save_dict(self):
        """
        make easier for saving model when need save additional arguments
        """
        # base arguments for all models
        # save_dict = {
        #     'model': self.model.state_dict(),
        #     'ema_model': self.ema_model.state_dict(),
        #     'optimizer': self.optimizer.state_dict(),
        #     'scheduler': self.scheduler.state_dict(),
        #     'loss_scaler': self.loss_scaler.state_dict(),
        #     'epoch': self.epoch + 1,
        #     'it': self.it + 1,
        #     'best_it': self.best_it,
        #     'best_eval_acc': self.best_eval_acc,
        # }
        save_dict = {
            'model_1': self.model_1.state_dict(),
            'model_2': self.model_2.state_dict(),
            'ema_model_1': self.ema_model_1.state_dict(),
            'ema_model_2': self.ema_model_2.state_dict(),
            'optimizer_1': self.optimizer['model_1'].state_dict(),
            'optimizer_2': self.optimizer['model_2'].state_dict(),
            'scheduler_1': self.scheduler['model_1'].state_dict(),
            'scheduler_2': self.scheduler['model_2'].state_dict(),
            'loss_scaler_1': self.loss_scaler_1.state_dict(),
            'loss_scaler_2': self.loss_scaler_2.state_dict(),
            'epoch': self.epoch + 1,
            'it': self.it + 1,
            'best_it': self.best_it,
            'best_eval_acc': self.best_eval_acc,
        }
        return save_dict

    def save_model(self, save_name, save_path):
        """
        save model and specified parameters for resume
        """
        save_filename = os.path.join(save_path, save_name)
        save_dict = self.get_save_dict()
        torch.save(save_dict, save_filename)
        self.print_fn(f"model saved: {save_filename}")

    def load_model(self, load_path):
        """
        load model and specified parameters for resume
        """
        # checkpoint = torch.load(load_path, map_location='cpu')
        # self.model.load_state_dict(checkpoint['model'])
        # self.ema_model.load_state_dict(checkpoint['ema_model'])
        # self.optimizer.load_state_dict(checkpoint['optimizer'])
        # self.scheduler.load_state_dict(checkpoint['scheduler'])
        # self.loss_scaler.load_state_dict(checkpoint['loss_scaler'])
        # self.epoch = checkpoint['epoch']
        # self.it = checkpoint['it']
        # self.best_it = checkpoint['best_it']
        # self.best_eval_acc = checkpoint['best_eval_acc']
        # self.print_fn('model loaded')
        checkpoint = torch.load(load_path, map_location='cpu')
        self.model_1.load_state_dict(checkpoint['model_1'])
        self.model_2.load_state_dict(checkpoint['model_2'])
        self.ema_model_1.load_state_dict(checkpoint['ema_model_1'])
        self.ema_model_2.load_state_dict(checkpoint['ema_model_2'])
        self.optimizer['model_1'].load_state_dict(checkpoint['optimizer_1'])
        self.optimizer['model_2'].load_state_dict(checkpoint['optimizer_2'])
        self.scheduler['model_1'].load_state_dict(checkpoint['scheduler_1'])
        self.scheduler['model_2'].load_state_dict(checkpoint['scheduler_2'])
        self.loss_scaler_1.load_state_dict(checkpoint['loss_scaler_1'])
        self.loss_scaler_2.load_state_dict(checkpoint['loss_scaler_2'])
        self.epoch = checkpoint['epoch']
        self.it = checkpoint['it']
        self.best_it = checkpoint['best_it']
        self.best_eval_acc = checkpoint['best_eval_acc']
        self.print_fn('model loaded')
        return checkpoint

    def check_prefix_state_dict(self, state_dict):
        """
        remove prefix state dict in ema model
        """
        new_state_dict = dict()
        for key, item in state_dict.items():
            if key.startswith('module'):
                new_key = '.'.join(key.split('.')[1:])
            else:
                new_key = key
            new_state_dict[new_key] = item
        return new_state_dict

    def register_hook(self, hook, name=None, priority='LOWEST'):
        """
        Register a hook into the hook list.
        The hook will be inserted into a priority queue, with the specified
        priority (See :class:`Priority` for details of priorities).
        For hooks with the same priority, they will be triggered in the same
        order as they are registered.
        Args:
            hook (:obj:`Hook`): The hook to be registered.
            name (:str, default to None): Name of the hook to be registered. Default is the hook class name.
            priority (int or str or :obj:`Priority`): Hook priority.
                Lower value means higher priority.
        """
        assert isinstance(hook, Hook)
        if hasattr(hook, 'priority'):
            raise ValueError('"priority" is a reserved attribute for hooks')
        priority = get_priority(priority)
        hook.priority = priority  # type: ignore
        hook.name = name if name is not None else type(hook).__name__

        # insert the hook to a sorted list
        inserted = False
        for i in range(len(self._hooks) - 1, -1, -1):
            if priority >= self._hooks[i].priority:  # type: ignore
                self._hooks.insert(i + 1, hook)
                inserted = True
                break

        if not inserted:
            self._hooks.insert(0, hook)

        # call set hooks
        self.hooks_dict = OrderedDict()
        for hook in self._hooks:
            self.hooks_dict[hook.name] = hook

    def call_hook(self, fn_name, hook_name=None, *args, **kwargs):
        """Call all hooks.
        Args:
            fn_name (str): The function name in each hook to be called, such as
                "before_train_epoch".
            hook_name (str): The specific hook name to be called, such as
                "param_update" or "dist_align", uesed to call single hook in train_step.
        """

        if hook_name is not None:
            return getattr(self.hooks_dict[hook_name], fn_name)(self, *args, **kwargs)

        for hook in self.hooks_dict.values():
            if hasattr(hook, fn_name):
                getattr(hook, fn_name)(self, *args, **kwargs)

    def registered_hook(self, hook_name):
        """
        Check if a hook is registered
        """
        return hook_name in self.hooks_dict

    @staticmethod
    def get_argument():
        """
        Get specificed arguments into argparse for each algorithm
        """
        return {}
