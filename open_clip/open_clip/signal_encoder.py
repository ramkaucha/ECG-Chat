import torch.nn as nn
from functools import partial
from .model import CLIPEcgCfg
from .transformer import EcgTransformer, LayerNormFp32, LayerNorm, QuickGELU

class SignalEncoder(nn.Module):
    """
    Encodes raw ECGs into fixed length embeddings
    Input: Tensor [B, lead_num, seq_length]
    Output: Tensor [B, embedding_dim] + token embeddings if needed
    """
    def __init__(
        self,
        embed_dim: int,
        ecg_cfg: CLIPEcgCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        if isinstance(ecg_cfg, dict):
            ecg_cfg = CLIPEcgCfg(**ecg_cfg)

        act_layer = QuickGELU if quick_gelu else nn.GELU
        ecg_heads = ecg_cfg.width // ecg_cfg.head_width
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
        if ecg_cfg.norm_kwargs:
            norm_layer = partial(norm_layer, **ecg_cfg.norm_kwargs)
        if ecg_cfg.act_kwargs is not None:
            act_layer = partial(act_layer, **ecg_cfg.act_kwargs)
        
        self.model = EcgTransformer(
            seq_length=ecg_cfg.seq_length,
            patch_size=ecg_cfg.patch_size,
            lead_num=ecg_cfg.lead_num,
            width=ecg_cfg.width,
            layers=ecg_cfg.layers,
            heads=ecg_heads,
            mlp_ratio=ecg_cfg.mlp_ratio,
            ls_init_value=ecg_cfg.ls_init_value,
            patch_dropout=ecg_cfg.patch_dropout,
            attentional_pool=ecg_cfg.attentional_pool,
            attn_pooler_queries=ecg_cfg.attn_pooler_queries,
            attn_pooler_heads=ecg_cfg.attn_pooler_heads,
            pos_embed_type=ecg_cfg.pos_embed_type,
            no_ln_pre=ecg_cfg.no_ln_pre,
            final_ln_after_pool=ecg_cfg.final_ln_after_pool,
            pool_type=ecg_cfg.pool_type,
            output_tokens=ecg_cfg.output_tokens,
            output_dim=embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer
        )
        self.embed_dim = embed_dim
    
    def forward(self, ecgs: torch.Tensor):
        return self.model(ecgs)
