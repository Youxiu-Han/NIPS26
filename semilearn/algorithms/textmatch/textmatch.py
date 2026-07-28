import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from semilearn.core.algorithmbase import AlgorithmBase
from semilearn.algorithms.hooks import DistAlignQueueHook, PseudoLabelingHook, FixedThresholdingHook
from semilearn.algorithms.utils import ce_loss, consistency_loss, SSL_Argument, str2bool

from .utils import mb_sup_loss, cls_names_input
from .coop import *
from semilearn.datasets.cv_datasets import find_car_classnames, find_dog_classnames, find_cub_classnames, find_aircraft_classnames, find_aves_classnames, find_fungi_classnames


def supervised_contrastive_loss(features, labels, temperature=0.1):

    device = features.device
    batch_size = features.shape[0]
    
    if batch_size <= 1:
        return torch.tensor(0.0, device=device)
    
    # 归一化特征
    features = F.normalize(features, dim=1)
    
    # 计算相似度矩阵 [batch_size, batch_size]
    similarity_matrix = torch.matmul(features, features.T) / temperature
    
    # 创建标签掩码：同类为1，异类为0
    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)  # [batch_size, batch_size]
    
    # 移除对角线（自己和自己的相似度）
    logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
    mask = mask * logits_mask  # 移除对角线上的正样本
    
    # 计算log_softmax
    # 对每一行减去最大值，提高数值稳定性
    logits_max, _ = torch.max(similarity_matrix * logits_mask, dim=1, keepdim=True)
    logits = similarity_matrix - logits_max.detach()
    
    # 计算分母：所有非自己的样本
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-6)
    
    # 计算每个样本的正样本对数量
    mask_pos_pairs = mask.sum(dim=1)
    mask_pos_pairs = torch.clamp(mask_pos_pairs, min=1e-6)  # 避免除零
    
    # 计算损失：对每个anchor，平均其所有正样本对的loss
    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask_pos_pairs
    
    # 最终损失
    loss = -mean_log_prob_pos.mean()
    
    return loss


class TextMatchNet(nn.Module):
    def __init__(self, base, num_classes, proj_size=128, args=None, use_pl="", num_ctx=None, use_adapter=False):
        super(TextMatchNet, self).__init__()
        self.backbone = base
        self.feat_planes = base.num_features
        self.clip_feat_dim = 512  # CLIP ViT-B/16 特征维度

        self.mlp_proj = nn.Sequential(*[
            nn.Linear(self.feat_planes, self.feat_planes),
            nn.ReLU(inplace=False),
            nn.Linear(self.feat_planes, proj_size)
        ])

        self.mb_classifiers = nn.Linear(proj_size, num_classes * 2, bias=False)
        self.openset_classifier = nn.Linear(proj_size, num_classes + 1)

        nn.init.xavier_normal_(self.openset_classifier.weight.data)
        self.openset_classifier.bias.data.zero_()

        ####### coop backbone #######
        if args.dataset == "car196":
            cls_name = find_car_classnames(args)
        elif args.dataset == "dog120":
            cls_name = find_dog_classnames(args)
        elif args.dataset == "cub200_2011":
            cls_name = find_cub_classnames(args)
        elif args.dataset == "aircraft100":
            cls_name = find_aircraft_classnames(args)
        elif args.dataset == "semi_aves200":
            cls_name = find_aves_classnames(args)
        elif args.dataset == "semi_fungi":
            cls_name = find_fungi_classnames(args)
        print(f"Loading the classnames of labeled class {cls_name}")
        print(f"Loading CLIP (backbone: {args.coop_backbone})")
        clip_model = load_clip_to_cpu(args)
        if args.coop_prec == "fp32" or args.coop_prec == "amp":
            clip_model.float()
        print("Building custom CLIP")
        self.coop_model = CustomCLIP(args, cls_name, clip_model, use_pl, num_ctx, use_adapter)
        for name, param in self.coop_model.named_parameters():
            if "prompt_learner" in name or "img_adapter" in name or "class_cache" in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
        self.scaler = GradScaler() if args.coop_prec == "amp" else None
        self.topk = args.topk_patches  # top-k patch token选择
        
    def forward(self, x, labels=None, use_enhanced_text=False, update_cache=False, **kwargs):

        logits, token_index, image_features, aligned_features = self.coop_model(
            x, self.topk, 
            labels=labels,
            use_enhanced_text=use_enhanced_text,
            update_cache=update_cache
        )

        return_dict = {
            'logits': logits, 
            'token_index': token_index,
            'image_features': image_features,
            'aligned_features': aligned_features,
        }
        return return_dict

    def group_matcher(self, coarse=False):
        matcher = self.backbone.group_matcher(coarse, prefix='backbone.')
        return matcher
    
class TextMatch(AlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)
        # textmatch specified arguments
        self.dist_align = args.dist_align
        self.p_cutoff = args.p_cutoff
        self.q_cutoff = args.q_cutoff
        self.lambda_mb = args.mb_loss_ratio
        self.lambda_op = args.op_loss_ratio
        self.cache_warmup_iter = args.cache_warmup_iter  # cache warmup迭代数
        # 监督对比学习参数
        self.use_supcon = args.use_supcon  # 是否使用监督对比学习
        self.lambda_supcon = args.supcon_loss_ratio  # 对比学习损失权重
        self.supcon_temperature = args.supcon_temperature  # 温度参数
        # OOD合成参数
        self.use_ood_synthesis = args.use_ood_synthesis  # 是否使用OOD合成
        self.ood_mix_ratio = args.ood_mix_ratio  # 混合比例
        self.lambda_ood = args.ood_loss_ratio  # OOD损失权重
        
        # 文本侧冲突参数
        self.use_text_conflict = args.use_text_conflict  # 是否使用文本侧冲突
        self.lambda_text_conflict = args.text_conflict_loss_ratio  # 文本冲突损失权重
        self.text_conflict_temperature = args.text_conflict_temperature  # 对比损失温度
        
        # ===== 保存关键参数到txt文件 =====
        self._save_key_params(args)
    
    def _save_key_params(self, args):
        """保存关键实验参数到txt文件，方便后续对比"""
        import os
        from datetime import datetime
        
        # 保存路径
        save_path = os.path.join(args.save_dir, args.save_name, 'key_params.txt')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 关键参数
        key_params = {
            'seed': args.seed,
            'use_supcon': args.use_supcon,
            'supcon_loss_ratio': args.supcon_loss_ratio,
            'supcon_temperature': args.supcon_temperature,
            'p_cutoff': args.p_cutoff,
            'ulb_loss_ratio': args.ulb_loss_ratio,
            'cache_warmup_iter': args.cache_warmup_iter,
            'cache_alpha': args.cache_alpha,  # cache融合权重
            'cache_momentum': args.cache_momentum,  # cache动量系数
            'use_feature_align': args.use_feature_align,  # 是否使用feature_align（消融实验）
            'topk_patches': args.topk_patches,  # top-k patch token选择数量
            'da_len': args.da_len,
            # OOD合成参数
            'use_ood_synthesis': args.use_ood_synthesis,
            'ood_mix_ratio': args.ood_mix_ratio,
            'ood_loss_ratio': args.ood_loss_ratio,
            # 文本侧冲突参数
            'use_text_conflict': args.use_text_conflict,
            'text_conflict_loss_ratio': args.text_conflict_loss_ratio,
            'text_conflict_temperature': args.text_conflict_temperature,
        }
        
        with open(save_path, 'w') as f:
            f.write(f"===== 关键实验参数 =====\n")
            f.write(f"保存时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"实验名称: {args.save_name}\n")
            f.write(f"\n")
            for k, v in key_params.items():
                f.write(f"{k}: {v}\n")
        
        self.print_fn(f"关键参数已保存到: {save_path}")

    def set_hooks(self):
        self.register_hook(
            DistAlignQueueHook(num_classes=self.num_classes, queue_length=self.args.da_len, p_target_type='uniform'),
            "DistAlignHook"
        )
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FixedThresholdingHook(), "MaskingHook")
        super().set_hooks()

    def _compute_ood_synthesis_loss(self, image_features, labels, model):
      
        B, D = image_features.shape
        num_classes = self.num_classes
        
        # 1. 获取class cache（model_1和model_2共享cache，所以用哪个都一样）
        # 但为了一致性，从传入的model获取
        class_cache = model.coop_model.class_cache.cache.detach()  # [num_classes, dim], no_grad
        
        # 确保数据类型一致
        cache_dtype = class_cache.dtype
        image_features = image_features.to(cache_dtype)
        
        # 2. 用cache中其他类的特征"污染"当前样本的image_features
        synthetic_features = []
        for i in range(B):
            y_i = labels[i].item()
            # 随机选择一个不同的类
            other_classes = [c for c in range(num_classes) if c != y_i]
            sampled_class = random.choice(other_classes)
            
            # FeatureMix: β * image_features[i] + (1-β) * cache[other_class]
            # image_features[i] 有梯度，cache[other_class] 无梯度但提供OOD方向
            beta = 1.0 - self.ood_mix_ratio
            mixed_feature = beta * image_features[i] + (1 - beta) * class_cache[sampled_class]
            synthetic_features.append(mixed_feature)
        
        synthetic_features = torch.stack(synthetic_features, dim=0)  # [B, dim]
        
        # 3. 归一化（保留梯度）
        synthetic_features = F.normalize(synthetic_features.float(), dim=-1)
        
        # 4. 获取text_features并计算logits（使用传入的model，确保一致性）
        prompts, _ = model.coop_model.prompt_learner()
        text_features = model.coop_model.text_encoder(
            prompts, model.coop_model.tokenized_prompts
        )
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        logit_scale = model.coop_model.logit_scale.exp()
        logits_syn = logit_scale * synthetic_features @ text_features.t().float()  # [B, num_classes]
        
        # 5. Uniform Loss：希望合成OOD的分布接近均匀分布
        # L_oe = -mean(log(p)) = log(C) - H(p)，等价于最大化熵
        probs_syn = F.softmax(logits_syn, dim=-1)
        uniform_target = torch.ones_like(probs_syn) / num_classes  # [B, num_classes]
        
        # KL散度到均匀分布
        ood_loss = -torch.mean(torch.sum(uniform_target * torch.log(probs_syn + 1e-8), dim=-1))
        
        return ood_loss

    def _compute_text_conflict_loss(self, image_features, labels, model):
        """
        对比式文本冲突损失：
        - 正样本：text[y] + cache[y] → 拉近
        - 负样本：text[y] + cache[d] → 推开
        
        使用 InfoNCE 风格的对比损失，让 image 更接近正确的 cache-enhanced text，
        同时远离语义冲突的 text。
        
        关键：text特征和cache做detach，梯度只回传到img_adapter和fusion（θh）
        """
        B, D = image_features.shape
        num_classes = self.num_classes
        
        # 确保数据类型一致
        cache_dtype = model.coop_model.class_cache.cache.dtype
        image_features = image_features.to(cache_dtype)
        
        # 只对text特征和cache做detach，fusion保留梯度（LPC需要优化θh）
        with torch.no_grad():
            # 1. 获取原始text特征 [C, D]
            prompts, _ = model.coop_model.prompt_learner()
            text_features = model.coop_model.text_encoder(prompts, model.coop_model.tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # 2. 获取cache
            cache = model.coop_model.class_cache.cache  # [C, D]
        
        fusion = model.coop_model.class_cache.fusion
        alpha = model.coop_model.class_cache.alpha
        
        # 3. 为每个样本构造正确的text和冲突的text
        correct_text_features = []
        conflict_text_features = []
        
        for i in range(B):
            y_i = labels[i].item()
            text_y = text_features[y_i]  # [D]
            cache_y = cache[y_i]  # [D]
            
            # 正确的text：text[y] + cache[y]
            concat_correct = torch.cat([text_y, cache_y], dim=-1)
            fused_correct = fusion(concat_correct.unsqueeze(0)).squeeze(0)
            correct_text = alpha * fused_correct + text_y
            correct_text = correct_text / correct_text.norm(dim=-1, keepdim=True)
            correct_text_features.append(correct_text)
            
            # 冲突的text：text[y] + cache[d]
            other_classes = [c for c in range(num_classes) if c != y_i]
            conflict_class = random.choice(other_classes)
            cache_d = cache[conflict_class]
            
            concat_conflict = torch.cat([text_y, cache_d], dim=-1)
            fused_conflict = fusion(concat_conflict.unsqueeze(0)).squeeze(0)
            conflict_text = alpha * fused_conflict + text_y
            conflict_text = conflict_text / conflict_text.norm(dim=-1, keepdim=True)
            conflict_text_features.append(conflict_text)
        
        correct_text_features = torch.stack(correct_text_features, dim=0)  # [B, D]
        conflict_text_features = torch.stack(conflict_text_features, dim=0)  # [B, D]
        
        # 4. 计算相似度
        image_features_norm = F.normalize(image_features.float(), dim=-1)
        
        # 正样本相似度：image 与 correct_text
        sim_pos = (image_features_norm * correct_text_features.float()).sum(dim=-1)  # [B]
        
        # 负样本相似度：image 与 conflict_text
        sim_neg = (image_features_norm * conflict_text_features.float()).sum(dim=-1)  # [B]
        
        # 使用温度参数（而不是 logit_scale，避免过大的缩放）
        logits = torch.stack([sim_pos, sim_neg], dim=-1) / self.text_conflict_temperature  # [B, 2]
        labels_contrast = torch.zeros(B, dtype=torch.long, device=image_features.device)
        loss = F.cross_entropy(logits, labels_contrast)
        
        return loss

    def set_model(self):
        model = super().set_model()  # backbone
        model_dict={
            'model_1': TextMatchNet(model, num_classes=self.num_classes, args=self.args, use_pl="", num_ctx=16, use_adapter=True),
            'model_2': TextMatchNet(model, num_classes=self.num_classes, args=self.args, use_pl="", num_ctx=16, use_adapter=True)
            }
        # 让 Model 2 共享 Model 1 的 cache（Model 1 更新 cache，Model 2 使用增强文本）
        model_dict['model_2'].coop_model.class_cache = model_dict['model_1'].coop_model.class_cache
        return model_dict

    def set_ema_model(self):
        ema_model = self.net_builder(num_classes=self.num_classes)
        ema_model_1 = TextMatchNet(ema_model, num_classes=self.num_classes, args=self.args, use_pl="", num_ctx=16, use_adapter=True)
        ema_model_2 = TextMatchNet(ema_model, num_classes=self.num_classes, args=self.args, use_pl="", num_ctx=16, use_adapter=True)
        ema_model_1.load_state_dict(self.model_1.state_dict())
        ema_model_2.load_state_dict(self.model_2.state_dict())
        # 让 EMA Model 2 共享 EMA Model 1 的 cache
        ema_model_2.coop_model.class_cache = ema_model_1.coop_model.class_cache
        ema_model_dict = {
            'ema_model_1': ema_model_1, 
            'ema_model_2': ema_model_2
            }
        return ema_model_dict

    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s, y_ulb=None):
        num_lb = y_lb.shape[0]
        num_ulb = x_ulb_w.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                # 构造labels：有标签数据用真实标签，无标签数据用-1
                labels_all = torch.cat([y_lb, torch.full((num_ulb * 2,), -1, dtype=torch.long, device=y_lb.device)])
                labels_lb = y_lb  # 只有标签数据的labels
                
                # ===== 渐进式Cache更新策略 =====
                # Phase 1 (0 ~ cache_warmup_iter): 只用有标签数据更新cache
                # Phase 2 (cache_warmup_iter+): 引入高置信度无标签样本更新cache
                labels_for_cache = labels_all.clone()
                
                if self.it >= self.cache_warmup_iter:
                    # Warmup之后，用高置信度无标签样本的伪标签更新cache
                    # 先做一次前向传播获取无标签数据的预测
                    with torch.no_grad():
                        # 只对无标签weak augmentation部分做预测
                        ulb_w_inputs = x_ulb_w
                        temp_outputs = self.model_1(ulb_w_inputs, labels=None,
                                                    use_enhanced_text=False, update_cache=False)
                        probs_ulb = F.softmax(temp_outputs['logits'], dim=-1)
                        max_probs, pseudo_labels = probs_ulb.max(dim=-1)
                        
                        # 高置信度样本（置信度 > p_cutoff）
                        high_conf_mask = max_probs > self.p_cutoff
                        
                        # 更新labels_for_cache：将高置信度无标签样本的位置用伪标签替换
                        # labels_for_cache结构: [有标签(num_lb), 无标签weak(num_ulb), 无标签strong(num_ulb)]
                        ulb_w_start = num_lb
                        ulb_w_end = num_lb + num_ulb
                        
                        # 只对weak augmentation的高置信度样本更新伪标签
                        labels_for_cache[ulb_w_start:ulb_w_end][high_conf_mask] = pseudo_labels[high_conf_mask]
                
                # Model 1: 使用原始文本，更新cache
                outputs_1 = self.model_1(
                    inputs, 
                    None,  
                    #labels=labels_all,  # 只用有标签样本更新
                    labels=labels_for_cache,  # 用有标签样本和高置信度更新cache
                    use_enhanced_text=False,  # 使用原始COOP文本
                    update_cache=True  # 更新cache（内部会判断是否有标签>=0）
                )
                
                # Model 2: 使用增强文本，不更新cache
                outputs_2 = self.model_2(
                    inputs, 
                    None,
                    labels=None,  # 不需要labels（不做类级别选择）
                    use_enhanced_text=True,  # 使用cache增强的文本
                    update_cache=False  # 不更新cache
                )
                
                # labeled data
                logits_x_lb_1 = outputs_1['logits'][:num_lb]
                logits_x_lb_2 = outputs_2['logits'][:num_lb]
                # ulabeled data
                logits_x_ulb_w_1, logits_x_ulb_s_1 = outputs_1['logits'][num_lb:].chunk(2)
                logits_x_ulb_w_2, logits_x_ulb_s_2 = outputs_2['logits'][num_lb:].chunk(2)
            else:
                raise ValueError("Bad configuration: use_cat should be True!")

            # supervised losses
            sup_closed_loss_1 = F.cross_entropy(logits_x_lb_1, y_lb)
            sup_closed_loss_2 = F.cross_entropy(logits_x_lb_2, y_lb)
            sup_loss = sup_closed_loss_1 + sup_closed_loss_2

            # generator closed-set and open-set targets (pseudo-labels)
            with torch.no_grad():
                p_1 = F.softmax(logits_x_ulb_w_1, dim=-1)
                targets_p_1 = p_1.detach()
                p_2 = F.softmax(logits_x_ulb_w_2, dim=-1)
                targets_p_2 = p_2.detach()

                if self.dist_align:
                    targets_p_1 = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=targets_p_1)
                    targets_p_2 = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=targets_p_2)

            p_mask_1 = self.call_hook("masking", "MaskingHook", cutoff=self.p_cutoff,
                                        logits_x_ulb=targets_p_1, softmax_x_ulb=False)
            p_mask_2 = self.call_hook("masking", "MaskingHook", cutoff=self.p_cutoff,logits_x_ulb=targets_p_2, softmax_x_ulb=False)

            ui_loss_1 = consistency_loss(logits_x_ulb_s_1, targets_p_1, 'ce', mask=p_mask_1)
            ui_loss_2 = consistency_loss(logits_x_ulb_s_2, targets_p_2, 'ce', mask=p_mask_2)

            # ===== 监督对比学习损失（SupCon Loss）=====
            if self.use_supcon:
                image_features_lb_1 = outputs_1['aligned_features'][:num_lb]
                image_features_lb_2 = outputs_2['image_features'][:num_lb]
                
                image_features_ulb_w_1 = outputs_1['aligned_features'][num_lb:num_lb+num_ulb]
                image_features_ulb_w_2 = outputs_2['image_features'][num_lb:num_lb+num_ulb]
                
                with torch.no_grad():
                    pseudo_labels_1 = targets_p_1.argmax(dim=-1)
                    pseudo_labels_2 = targets_p_2.argmax(dim=-1)
                
                p_mask_1 = p_mask_1.bool()
                p_mask_2 = p_mask_2.bool()

                high_conf_features_1 = image_features_ulb_w_1[p_mask_1]
                high_conf_labels_1 = pseudo_labels_1[p_mask_1]
                combined_features_1 = torch.cat([image_features_lb_1, high_conf_features_1], dim=0)
                combined_labels_1 = torch.cat([y_lb, high_conf_labels_1], dim=0)
                
                high_conf_features_2 = image_features_ulb_w_2[p_mask_2]
                high_conf_labels_2 = pseudo_labels_2[p_mask_2]
                combined_features_2 = torch.cat([image_features_lb_2, high_conf_features_2], dim=0)
                combined_labels_2 = torch.cat([y_lb, high_conf_labels_2], dim=0)
                
                supcon_loss_1 = supervised_contrastive_loss(
                    combined_features_1.float(), combined_labels_1, temperature=self.supcon_temperature
                )
                supcon_loss_2 = supervised_contrastive_loss(
                    combined_features_2.float(), combined_labels_2, temperature=self.supcon_temperature
                )
            else:
                supcon_loss_1 = torch.tensor(0.0).to(self.gpu)
                supcon_loss_2 = torch.tensor(0.0).to(self.gpu)

            # ===== OOD合成损失（Cache-driven Outlier Synthesis）=====
            # 两个模型都学习OOD边界，协同增强OOD过滤能力
            if self.use_ood_synthesis:
                ood_synth_loss_1 = torch.tensor(0.0).to(self.gpu)
                ood_synth_loss_2 = self._compute_ood_synthesis_loss(
                    outputs_2['image_features'][:num_lb],  # Model 2的image features
                    y_lb,
                    self.model_2,  # 使用Model 2的text_encoder
                )
            else:
                ood_synth_loss_1 = torch.tensor(0.0).to(self.gpu)
                ood_synth_loss_2 = torch.tensor(0.0).to(self.gpu)

            # ===== 文本侧冲突损失（Text Conflict Loss）=====
            # 用其他类的cache"污染"当前类的prompt，训练img_adapter远离语义冲突的描述
            if self.use_text_conflict:
                text_conflict_loss_1 = torch.tensor(0.0).to(self.gpu)
                text_conflict_loss_2 = self._compute_text_conflict_loss(
                    outputs_2['image_features'][:num_lb],
                    y_lb,
                    self.model_2,
                )
            else:
                text_conflict_loss_1 = torch.tensor(0.0).to(self.gpu)
                text_conflict_loss_2 = torch.tensor(0.0).to(self.gpu)

            total_loss = {
                'model_1': sup_closed_loss_1 + self.lambda_u * ui_loss_1 + self.lambda_supcon * supcon_loss_1,
                'model_2': sup_closed_loss_2 + self.lambda_u * ui_loss_2 + self.lambda_supcon * supcon_loss_2 + self.lambda_ood * ood_synth_loss_2 + self.lambda_text_conflict * text_conflict_loss_2
                }

        self.call_hook("param_update", "ParamUpdateHook", loss=total_loss)

        # ===== 验证class_cache是否起作用的指标 =====
        # 1. cache更新次数：有多少类被更新过
        cache_count = self.model_1.coop_model.class_cache.count
        cache_updated_classes = (cache_count > 0).sum().item()  # 被更新过的类别数
        cache_total_updates = cache_count.sum().item()  # 总更新次数
        
        # 3. OOD过滤统计（如果有真实标签y_ulb）
        ood_filtered_ratio_1 = 0.0
        ood_filtered_ratio_2 = 0.0
        id_kept_ratio_1 = 0.0
        id_kept_ratio_2 = 0.0
        if y_ulb is not None:
            with torch.no_grad():
                # 判断哪些是真实OOD（标签 < 0 或 >= num_classes）
                is_ood = (y_ulb < 0) | (y_ulb >= self.num_classes)  # [num_ulb]
                is_id = ~is_ood
                num_ood = is_ood.sum().item()
                num_id = is_id.sum().item()
                
                if num_ood > 0:
                    ood_filtered_ratio_1 = ((~p_mask_1.bool()) & is_ood).sum().item() / num_ood
                    ood_filtered_ratio_2 = ((~p_mask_2.bool()) & is_ood).sum().item() / num_ood
                if num_id > 0:
                    id_kept_ratio_1 = (p_mask_1.bool() & is_id).sum().item() / num_id
                    id_kept_ratio_2 = (p_mask_2.bool() & is_id).sum().item() / num_id
        
        tb_dict = {
            'train/model_1_loss': total_loss['model_1'].item(),
            'train/model_2_loss': total_loss['model_2'].item(),
            'train/sup_loss': sup_loss.item(),
            'train/sup_closed_loss_1': sup_closed_loss_1.item(),
            'train/sup_closed_loss_2': sup_closed_loss_2.item(),
            'train/ui_loss_1': ui_loss_1.item(),
            'train/ui_loss_2': ui_loss_2.item(),
            'train/p_mask_ratio_1': p_mask_1.float().mean().item(),
            'train/p_mask_ratio_2': p_mask_2.float().mean().item(),
            'train/supcon_loss_1': supcon_loss_1.item(),
            'train/supcon_loss_2': supcon_loss_2.item(),
            'train/ood_synth_loss_1': ood_synth_loss_1.item(),
            'train/ood_synth_loss_2': ood_synth_loss_2.item(),
            'train/text_conflict_loss_1': text_conflict_loss_1.item(),
            'train/text_conflict_loss_2': text_conflict_loss_2.item(),
            'train/cache_updated_classes': cache_updated_classes,
            'train/cache_total_updates': cache_total_updates,
            'train/ood_filtered_ratio_1': ood_filtered_ratio_1,
            'train/ood_filtered_ratio_2': ood_filtered_ratio_2,
            'train/id_kept_ratio_1': id_kept_ratio_1,
            'train/id_kept_ratio_2': id_kept_ratio_2,
        }

        return tb_dict

    def get_save_dict(self):
        save_dict = super().get_save_dict()
        save_dict['p_model'] = self.hooks_dict['DistAlignHook'].p_model.cpu()
        save_dict['p_model_ptr'] = self.hooks_dict['DistAlignHook'].p_model_ptr.cpu()
        return save_dict

    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.hooks_dict['DistAlignHook'].p_model = checkpoint['p_model'].cuda(self.args.gpu)
        self.hooks_dict['DistAlignHook'].p_model_ptr = checkpoint['p_model_ptr'].cuda(self.args.gpu)
        return checkpoint

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--dist_align', str2bool, True),
            SSL_Argument('--cache_warmup_iter', int, 10000),  # cache warmup迭代数，之后引入高置信度无标签样本
            # Cache模块超参数
            SSL_Argument('--cache_alpha', float, 0.2),  # 融合权重，控制cache特征与文本特征的融合比例
            SSL_Argument('--cache_momentum', float, 0.8),  # 动量系数，控制cache更新的稳定性
            SSL_Argument('--use_feature_align', str2bool, True),  # 是否使用feature_align模块（消融实验）
            SSL_Argument('--topk_patches', int, 5),  # top-k patch token选择数量（敏感性实验）
            # 监督对比学习参数
            SSL_Argument('--use_supcon', str2bool, False),  # 是否使用监督对比学习
            SSL_Argument('--supcon_loss_ratio', float, 0.1),  # 对比学习损失权重
            SSL_Argument('--supcon_temperature', float, 0.1),  # 温度参数，越小分布越尖锐
            # OOD合成参数（Cache-driven Outlier Synthesis）
            SSL_Argument('--use_ood_synthesis', str2bool, False),  # 是否使用OOD合成
            SSL_Argument('--ood_mix_ratio', float, 0.5),  # 混合比例，越大合成样本越像OOD
            SSL_Argument('--ood_loss_ratio', float, 0.1),  # OOD损失权重
            # 文本侧冲突参数（Text Conflict Loss）
            SSL_Argument('--use_text_conflict', str2bool, True),  # 是否使用文本侧冲突
            SSL_Argument('--text_conflict_loss_ratio', float, 1.0),  # 文本冲突损失权重
            SSL_Argument('--text_conflict_temperature', float, 0.1),  # 对比损失温度，越小越严格
        ]
