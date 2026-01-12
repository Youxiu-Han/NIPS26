import torch
import torch.nn as nn
import torch.nn.functional as F
import os
# CoOp
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
import numpy as np

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()


class ClassWiseCache(nn.Module):
    """
    类级别Cache + 特征对齐 + 特征融合
    """
    def __init__(self, num_classes, feature_dim=512, alpha=0.2, momentum=0.8, use_feature_align=True):
        super().__init__()
        
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.alpha = alpha  # 
        self.momentum = momentum 
        self.use_feature_align = use_feature_align  
        
        # 初始化cache（每个类一个entry，不需要梯度）
        self.register_buffer(
            'cache', 
            F.normalize(torch.randn(num_classes, feature_dim), dim=-1)
        )
        self.register_buffer('count', torch.zeros(num_classes))
        
        # 特征对齐模块（将视觉token对齐到文本空间）
        # 使用LayerNorm代替BatchNorm，避免小batch size下的不稳定问题
        self.feature_align = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),  # LayerNorm不依赖batch统计量
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # 特征融合模块（拼接 -> 线性变换）
        self.fusion = nn.Linear(2 * feature_dim, feature_dim, bias=False)
    
    def update(self, class_id, visual_features):
        """
        更新指定类的cache
        Args:
            class_id: int, 类别ID
            visual_features: [topk, dim], 该类样本的top-k patches
        """
        # 1. 对齐视觉特征到文本空间（可选，用于消融实验）
        if self.use_feature_align:
            aligned_features = self.feature_align(visual_features)  # [topk, dim]
        else:
            aligned_features = visual_features  # 直接使用原始特征
        
        # 2. 聚合该样本的top-k features
        aggregated = aligned_features.mean(dim=0)  # [dim]
        aggregated = F.normalize(aggregated, dim=-1)
        
        # 3. Momentum更新cache
        with torch.no_grad():
            if self.count[class_id] == 0:
                # 第一次见到这个类，直接赋值
                self.cache[class_id] = aggregated
            else:
                # 动量更新
                self.cache[class_id] = (
                    self.momentum * self.cache[class_id] + 
                    (1 - self.momentum) * aggregated
                )
                # 重新归一化
                self.cache[class_id] = F.normalize(self.cache[class_id], dim=-1)
            
            self.count[class_id] += 1
    
    def get_enhanced_text(self, text_features):
        """
        用cache增强文本特征（类别一一对应版本）
        
        Args:
            text_features: [num_classes, dim], 原始COOP文本特征
        Returns:
            enhanced_text: [num_classes, dim], 增强后的文本特征
        """
        visual_context = self.cache  # [C, dim]
        
        # 2. 融合：将文本特征和对应类别的视觉特征拼接后融合
        concat_features = torch.cat([text_features, visual_context], dim=-1)  # [C, 2*dim]
        fused = self.fusion(concat_features)  # [C, dim]
        
        # 3. 残差连接：enhanced = α * fused + original
        enhanced = self.alpha * fused + text_features
        
        return enhanced


def load_clip_to_cpu(args):
    backbone_name = args.coop_backbone
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x, k, q, v = self.transformer(x, None)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class PromptLearner(nn.Module):

    def __init__(self, args, classnames, clip_model, use_pl, num_ctx):
        super().__init__()
        n_cls = len(classnames)
        # n_ctx = args.n_ctx
        n_ctx = num_ctx
        ctx_init = use_pl
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = args.input_size
        
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            if args.csc:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        self.text_projection = nn.Linear(ctx_dim, 768)
        self.text_projection.half()

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = args.class_token_position

        

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
      
        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts, self.text_projection(self.ctx)

class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class CustomCLIP(nn.Module):
    def __init__(self, args, classnames, clip_model, use_pl="", num_ctx=None, use_adapter=False):
        super().__init__()
        self.prompt_learner = PromptLearner(args, classnames, clip_model, use_pl, num_ctx)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.use_adapter = use_adapter
        self.num_classes = len(classnames)
        
        self.img_adapter = nn.Sequential(*[
            nn.Linear(512, 512 // 4),
            nn.ReLU(inplace=False),
            nn.Linear(512 // 4, 512)
            ]).to(clip_model.dtype)
        
        # 添加类级别Cache模块 - 从args读取超参数
        cache_alpha = getattr(args, 'cache_alpha', 0.2)  # 默认0.2
        cache_momentum = getattr(args, 'cache_momentum', 0.8)  # 默认0.8
        use_feature_align = getattr(args, 'use_feature_align', True)  # 默认True（消融实验）
        self.class_cache = ClassWiseCache(
            num_classes=len(classnames),
            feature_dim=512,
            alpha=cache_alpha,
            momentum=cache_momentum,
            use_feature_align=use_feature_align
        )
        self.class_cache = self.class_cache.to(clip_model.dtype)
      
    def forward(self, image, embedded_feat, topk, labels=None, use_enhanced_text=False, update_cache=False):
        logit_scale = self.logit_scale.exp()

        # 1. 获取文本特征
        prompts, share_ctx = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts) # [C, 512]
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # 2. 如果使用增强文本（Model 2）
        if use_enhanced_text:
            text_features = self.class_cache.get_enhanced_text(text_features)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # 3. 提取图像特征
        image_features, token_features = self.image_encoder(image.type(self.dtype), embedded_feat)
        
        # 4. Adapter
        if self.use_adapter: 
            image_features = self.img_adapter(image_features) + image_features
            token_features = self.img_adapter(token_features) + token_features
        
        # 5. 归一化
        img_norm_features = image_features / image_features.norm(dim=-1, keepdim=True)
        token_norm_features = token_features / token_features.norm(dim=-1, keepdim=True)
        
        # 6. 计算相似度
        similarity = token_norm_features @ text_features.t() # [B, num_tokens, num_classes]
        similarity = torch.softmax(similarity, dim=-1)
        similarity = similarity.view(similarity.shape[0], -1, similarity.shape[-1])
        token_features = token_features.view(token_features.shape[0], -1, token_features.shape[-1])
        
        # 7. 选择top-k patches（类级别）
        batch_size = token_features.shape[0]
        
        # 判断是否有有效标签（labels >= 0）
        has_valid_labels = labels is not None and self.training
        
        if has_valid_labels:
            # 创建有效标签mask（-1表示无标签数据）
            valid_mask = (labels >= 0)  # [batch_size]
            
            # 对有标签数据：类级别token选择
            # 对无标签数据：使用最大相似度方式
            class_similarity = torch.zeros(batch_size, similarity.shape[1], device=similarity.device)
            
            for i in range(batch_size):
                if valid_mask[i]:
                    # 有标签数据：只看真实类别的相似度
                    class_similarity[i] = similarity[i, :, labels[i]]
                else:
                    # 无标签数据：使用最大相似度
                    class_similarity[i] = similarity[i].max(dim=-1)[0]
            
            # 选择top-k
            max_token_similarity, max_token_idx = torch.topk(class_similarity, topk, dim=-1)
        else:
            # 选择与任意类别最相关的top-k
            max_similarity, max_idx = torch.max(similarity, dim=-1)  # [B, num_tokens]
            max_token_similarity, max_token_idx = torch.topk(max_similarity, topk, dim=-1)
            valid_mask = None
        
        # 8. 收集选中的tokens
        token_selected = []
        for i in range(batch_size):
            tmep = token_features[i][max_token_idx[i]]
            token_selected.append(tmep)
        token_selected = torch.stack(token_selected, dim=0)  # [B, topk, 512]
        
        # 9. 更新cache（只在Model 1训练时，且只更新有标签样本！）
        if update_cache and has_valid_labels:
            for i in range(batch_size):
                if valid_mask[i]:  # 只更新有标签的样本！
                    class_id = labels[i].item()
                    # 传递该样本的top-k tokens给cache更新
                    self.class_cache.update(class_id, token_selected[i])
        
        # 10. 计算logits
        logits = logit_scale * img_norm_features @ text_features.t()
        
        # 返回image_features（经过adapter处理）供对比学习使用
        return logits, max_token_idx, image_features