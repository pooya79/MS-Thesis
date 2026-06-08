"""Standalone FastConformer encoder — weight-compatible port of NeMo's
ConformerEncoder for the non-streaming, rel_pos, regular-attention case
(the configuration used by nvidia/stt_fa_fastconformer_hybrid_large).

Submodule and parameter names match NeMo exactly so a pretrained
state_dict loads with strict=True. References:
  nemo/collections/asr/modules/conformer_encoder.py
  nemo/collections/asr/parts/submodules/conformer_modules.py
  nemo/collections/asr/parts/submodules/multi_head_attention.py
  nemo/collections/asr/parts/submodules/subsampling.py

Not ported (irrelevant for this checkpoint / inference): streaming caches,
local/longformer attention, stochastic depth, adapters, reductions,
SDPA path, chunking.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

INF_VAL = 10000.0


class Swish(nn.SiLU):
    """NeMo's Swish == SiLU."""


# --------------------------------------------------------------------------- #
# Subsampling (dw_striding)
# --------------------------------------------------------------------------- #
def calc_length(lengths, all_paddings, kernel_size, stride, ceil_mode, repeat_num=1):
    add_pad = all_paddings - kernel_size
    for _ in range(repeat_num):
        lengths = torch.div(lengths.to(torch.float) + add_pad, stride) + 1.0
        lengths = torch.ceil(lengths) if ceil_mode else torch.floor(lengths)
    return lengths.to(torch.int)


class ConvSubsampling(nn.Module):
    """dw_striding subsampling. Conv module is a plain Sequential whose child
    indices match NeMo's MaskedConvSequential, so weights map 1:1."""

    def __init__(self, subsampling_factor, feat_in, feat_out, conv_channels):
        super().__init__()
        self._sampling_num = int(math.log(subsampling_factor, 2))
        self._stride = 2
        self._kernel_size = 3
        self._ceil_mode = False
        self._left_padding = (self._kernel_size - 1) // 2
        self._right_padding = (self._kernel_size - 1) // 2

        in_channels = 1
        layers = []
        # Layer 1: full conv
        layers.append(
            nn.Conv2d(in_channels, conv_channels, self._kernel_size, stride=self._stride, padding=self._left_padding)
        )
        in_channels = conv_channels
        layers.append(nn.ReLU(True))
        # remaining: depthwise + pointwise
        for _ in range(self._sampling_num - 1):
            layers.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    self._kernel_size,
                    stride=self._stride,
                    padding=self._left_padding,
                    groups=in_channels,
                )
            )
            layers.append(nn.Conv2d(in_channels, conv_channels, 1, stride=1, padding=0, groups=1))
            layers.append(nn.ReLU(True))
            in_channels = conv_channels

        in_length = torch.tensor(feat_in, dtype=torch.float)
        out_length = calc_length(
            in_length,
            self._left_padding + self._right_padding,
            self._kernel_size,
            self._stride,
            self._ceil_mode,
            self._sampling_num,
        )
        self.out = nn.Linear(conv_channels * int(out_length), feat_out)
        self.conv = nn.Sequential(*layers)

    @staticmethod
    def _mask(x, lengths):
        # x: (B, C, T, F) -> mask (B, 1, T, 1) broadcastable
        t = x.size(2)
        m = torch.arange(t, device=x.device).expand(x.size(0), t) < lengths.unsqueeze(1)
        return m.unsqueeze(1).unsqueeze(-1).to(x.dtype)

    def forward(self, x, lengths):
        # x: (B, 1, T, F). Replicates NeMo MaskedConvSequential: mask padded
        # time steps to zero before every conv layer so padding does not leak.
        cur = lengths.clone().float()
        mask = self._mask(x, cur.long())
        for layer in self.conv:
            x = x * mask
            x = layer(x)
            if isinstance(layer, nn.Conv2d) and layer.stride != (1, 1):
                p = layer.padding
                cur = torch.div(cur + p[0] + p[0] - layer.kernel_size[0], layer.stride[0], rounding_mode="floor") + 1
                mask = self._mask(x, cur.long())
        x = x * mask
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).reshape(b, t, -1))  # (B, T', feat_out)
        return x, cur.long()


# --------------------------------------------------------------------------- #
# Relative positional encoding (Transformer-XL)
# --------------------------------------------------------------------------- #
class RelPositionalEncoding(nn.Module):
    def __init__(self, d_model, xscale=None, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.xscale = xscale
        self.max_len = max_len
        self.pe = None

    def extend_pe(self, length, device, dtype):
        needed = 2 * length - 1
        if self.pe is not None and self.pe.size(1) >= needed:
            return
        positions = torch.arange(length - 1, -length, -1, dtype=torch.float32, device=device).unsqueeze(1)
        pe = torch.zeros(positions.size(0), self.d_model, device=device)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32, device=device) * -(math.log(INF_VAL) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        self.pe = pe.unsqueeze(0).to(dtype)

    def forward(self, x):
        self.extend_pe(x.size(1), x.device, x.dtype)
        if self.xscale:
            x = x * self.xscale
        input_len = x.size(1)
        center_pos = self.pe.size(1) // 2 + 1
        start_pos = center_pos - input_len
        end_pos = center_pos + input_len - 1
        pos_emb = self.pe[:, start_pos:end_pos]
        return x, pos_emb


# --------------------------------------------------------------------------- #
# Relative-position multi-head attention
# --------------------------------------------------------------------------- #
class RelPositionMultiHeadAttention(nn.Module):
    def __init__(self, n_head, n_feat, dropout_rate=0.0):
        super().__init__()
        assert n_feat % n_head == 0
        self.d_k = n_feat // n_head
        self.s_d_k = math.sqrt(self.d_k)
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        self.pos_bias_u = nn.Parameter(torch.zeros(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.zeros(self.h, self.d_k))
        self.dropout = nn.Dropout(dropout_rate)

    def forward_qkv(self, query, key, value):
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        return q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

    def rel_shift(self, x):
        b, h, qlen, pos_len = x.size()
        x = F.pad(x, pad=(1, 0))
        x = x.view(b, h, -1, qlen)
        x = x[:, :, 1:].view(b, h, qlen, pos_len)
        return x

    def forward_attention(self, value, scores, mask):
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask, -INF_VAL)
            attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
        else:
            attn = torch.softmax(scores, dim=-1)
        p_attn = self.dropout(attn)
        x = torch.matmul(p_attn, value)
        x = x.transpose(1, 2).reshape(n_batch, -1, self.h * self.d_k)
        return self.linear_out(x)

    def forward(self, query, key, value, mask, pos_emb):
        q, k, v = self.forward_qkv(query, key, value)
        q = q.transpose(1, 2)  # (B, T, H, d_k)

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k).transpose(1, 2)

        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)

        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))
        matrix_bd = matrix_bd[:, :, :, : matrix_ac.size(-1)]
        scores = (matrix_ac + matrix_bd) / self.s_d_k
        return self.forward_attention(v, scores, mask)


# --------------------------------------------------------------------------- #
# Conformer sub-blocks
# --------------------------------------------------------------------------- #
class ConformerFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = Swish()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class ConformerConvolution(nn.Module):
    def __init__(self, d_model, kernel_size, norm_type="batch_norm"):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0
        padding = (kernel_size - 1) // 2
        self.pointwise_conv1 = nn.Conv1d(d_model, d_model * 2, 1)
        # depthwise conv: CausalConv1D with symmetric padding == plain Conv1d
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size, padding=padding, groups=d_model
        )
        if norm_type == "batch_norm":
            self.batch_norm = nn.BatchNorm1d(d_model)
        elif norm_type == "layer_norm":
            self.batch_norm = nn.LayerNorm(d_model)
        else:
            raise NotImplementedError(norm_type)
        self.norm_type = norm_type
        self.activation = Swish()
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, 1)

    def forward(self, x, pad_mask=None):
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = F.glu(x, dim=1)
        if pad_mask is not None:
            x = x.masked_fill(pad_mask.unsqueeze(1), 0.0)
        x = self.depthwise_conv(x)
        if self.norm_type == "layer_norm":
            x = self.batch_norm(x.transpose(1, 2)).transpose(1, 2)
        else:
            x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        return x.transpose(1, 2)


class ConformerLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, conv_kernel_size, conv_norm_type="batch_norm", dropout=0.0):
        super().__init__()
        self.fc_factor = 0.5
        self.norm_feed_forward1 = LayerNorm(d_model)
        self.feed_forward1 = ConformerFeedForward(d_model, d_ff, dropout)

        self.norm_conv = LayerNorm(d_model)
        self.conv = ConformerConvolution(d_model, conv_kernel_size, conv_norm_type)

        self.norm_self_att = LayerNorm(d_model)
        self.self_attn = RelPositionMultiHeadAttention(n_heads, d_model, dropout)

        self.norm_feed_forward2 = LayerNorm(d_model)
        self.feed_forward2 = ConformerFeedForward(d_model, d_ff, dropout)

        self.dropout = nn.Dropout(dropout)
        self.norm_out = LayerNorm(d_model)

    def forward(self, x, att_mask, pos_emb, pad_mask):
        residual = x
        x = self.norm_feed_forward1(x)
        x = self.feed_forward1(x)
        residual = residual + self.dropout(x) * self.fc_factor

        x = self.norm_self_att(residual)
        x = self.self_attn(query=x, key=x, value=x, mask=att_mask, pos_emb=pos_emb)
        residual = residual + self.dropout(x)

        x = self.norm_conv(residual)
        x = self.conv(x, pad_mask=pad_mask)
        residual = residual + self.dropout(x)

        x = self.norm_feed_forward2(residual)
        x = self.feed_forward2(x)
        residual = residual + self.dropout(x) * self.fc_factor

        return self.norm_out(residual)


# --------------------------------------------------------------------------- #
# Encoder
# --------------------------------------------------------------------------- #
class ConformerEncoder(nn.Module):
    def __init__(
        self,
        feat_in,
        n_layers,
        d_model,
        subsampling_factor=8,
        subsampling_conv_channels=None,
        ff_expansion_factor=4,
        n_heads=8,
        conv_kernel_size=9,
        conv_norm_type="batch_norm",
        xscaling=True,
        pos_emb_max_len=5000,
        att_context_size=(-1, -1),
    ):
        super().__init__()
        d_ff = d_model * ff_expansion_factor
        self.d_model = d_model
        self.xscale = math.sqrt(d_model) if xscaling else None
        self.att_context_size = list(att_context_size)
        if subsampling_conv_channels is None or subsampling_conv_channels == -1:
            subsampling_conv_channels = d_model

        self.pre_encode = ConvSubsampling(subsampling_factor, feat_in, d_model, subsampling_conv_channels)
        self.pos_enc = RelPositionalEncoding(d_model, xscale=self.xscale, max_len=pos_emb_max_len)

        self.layers = nn.ModuleList(
            [
                ConformerLayer(d_model, d_ff, n_heads, conv_kernel_size, conv_norm_type)
                for _ in range(n_layers)
            ]
        )
        self.out_proj = None  # feat_out == d_model for this checkpoint

    def _create_masks(self, padding_length, max_len, device):
        att_mask = torch.ones(1, max_len, max_len, dtype=torch.bool, device=device)
        if self.att_context_size[0] >= 0:
            att_mask = att_mask.triu(diagonal=-self.att_context_size[0])
        if self.att_context_size[1] >= 0:
            att_mask = att_mask.tril(diagonal=self.att_context_size[1])

        pad_mask = torch.arange(0, max_len, device=device).expand(padding_length.size(0), -1) < padding_length.unsqueeze(
            -1
        )
        pad_mask_for_att = pad_mask.unsqueeze(1).repeat(1, max_len, 1)
        pad_mask_for_att = torch.logical_and(pad_mask_for_att, pad_mask_for_att.transpose(1, 2))
        att_mask = torch.logical_and(pad_mask_for_att, att_mask)
        att_mask = ~att_mask
        pad_mask = ~pad_mask
        return pad_mask, att_mask

    def forward(self, audio_signal, length):
        """audio_signal: (B, feat_in, T); length: (B,). Returns (B, d_model, T'), (B,)."""
        audio_signal = audio_signal.transpose(1, 2)  # (B, T, feat_in)
        audio_signal = audio_signal.unsqueeze(1)  # (B, 1, T, feat_in) for conv2d subsampling
        audio_signal, length = self.pre_encode(audio_signal, length)

        max_len = audio_signal.size(1)
        audio_signal, pos_emb = self.pos_enc(audio_signal)
        pad_mask, att_mask = self._create_masks(length, max_len, audio_signal.device)

        for layer in self.layers:
            audio_signal = layer(audio_signal, att_mask=att_mask, pos_emb=pos_emb, pad_mask=pad_mask)

        if self.out_proj is not None:
            audio_signal = self.out_proj(audio_signal)

        return audio_signal.transpose(1, 2), length
