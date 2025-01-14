import math
import copy
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import RobertaTokenizerFast
from .text_encoder import RobertaModel, RobertaConfig
from apex.normalization import FusedLayerNorm
from colossalai.kernel.cuda_native.flash_attention import FlashAttention, FlashCrossAttention
from einops import rearrange


@torch.jit.script
def dropout_add(x: Tensor, residual: Tensor, prob: float, training: bool) -> Tensor:
    out = F.dropout(x, p=prob, training=training)
    out = residual + out
    return out


class DecoderEmbeddings(nn.Module):

    def __init__(self, vocab_size, hidden_dim, pad_token_id, max_position_embeddings, dropout):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_token_id)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_dim)

        self.LayerNorm = FusedLayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        input_shape = x.size()
        seq_length = input_shape[1]
        device = x.device

        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(input_shape)

        input_embeds = self.word_embeddings(x)
        position_embeds = self.position_embeddings(position_ids)

        embeddings = input_embeds + position_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class Transformer(nn.Module):

    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
        pass_pos_and_query=True,
        text_encoder_type="roberta-base",
        freeze_text_encoder=False,
        num_queries=200,
        max_decoding_step=256,
    ):
        super().__init__()

        self.max_decoding_step = max_decoding_step
        self.pass_pos_and_query = pass_pos_and_query
        self.norm = FusedLayerNorm(d_model)
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        encoder_norm = FusedLayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        decoder_norm = FusedLayerNorm(d_model) if normalize_before else None
        self.decoder = TransformerDecoder(decoder_layer,
                                          num_decoder_layers,
                                          decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.tokenizer = RobertaTokenizerFast.from_pretrained(text_encoder_type)
        config = RobertaConfig.from_pretrained(text_encoder_type)
        self.text_encoder = RobertaModel(config=config)

        self.embedding = DecoderEmbeddings(len(self.tokenizer) + num_queries + 2, d_model, 1, max_decoding_step, dropout)

        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)

        self.expander_dropout = 0.1
        self.resizer = FeatureResizer(
            input_feat_size=config.hidden_size,
            output_feat_size=d_model,
            dropout=self.expander_dropout,
        )

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src=None,
        mask=None,
        query_embed=None,
        pos_embed=None,
        text=None,
        encode_and_save=True,
        text_memory=None,
        img_memory=None,
        text_attention_mask=None,
        prev_indx=None,
    ):
        if encode_and_save:
            # flatten NxCxHxW to HWxNxC
            bs, c, h, w = src.shape
            src = src.flatten(2).permute(2, 0, 1)
            device = src.device
            pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
            mask = mask.flatten(1)

            if self.pass_pos_and_query:
                tgt = torch.zeros_like(query_embed)
            else:
                src, tgt, query_embed, pos_embed = src + 0.1 * pos_embed, query_embed, None, None

            device = src.device
            max_encoding_step = self.max_decoding_step

            if isinstance(text[0], str):
                # Encode the text
                tokenized = self.tokenizer.batch_encode_plus(text,
                                                             padding="max_length",
                                                             max_length=max_encoding_step,
                                                             truncation=True,
                                                             return_tensors="pt").to(device)
                encoded_text = self.text_encoder(**tokenized)

                # Transpose memory because pytorch's attention expects sequence first
                text_memory = encoded_text.last_hidden_state.transpose(0, 1)
                # Invert attention mask that we get from huggingface because its the opposite in pytorch transformer
                text_attention_mask = tokenized.attention_mask.ne(1).bool()

                # Resize the encoder hidden states to be of the same d_model as the decoder
                text_memory_resized = self.resizer(text_memory)
            elif isinstance(text[0], torch.Tensor):
                # The text is already encoded, use as is.
                text_attention_mask, text_memory_resized, tokenized = text
            ## added; support input tokenized in dataloader for unitab_pretrain
            elif len(text[0].keys()) == 2:
                tokenized = self.tokenizer.batch_encode_plus([''],
                                                             padding="max_length",
                                                             max_length=max_encoding_step,
                                                             truncation=True,
                                                             return_tensors="pt").to(device)
                tokenized['input_ids'] = torch.stack([text[ii]['input_ids'] for ii in range(len(text))],
                                                     dim=0).squeeze(1).to(device)[:, :tokenized['input_ids'].shape[-1]]
                tokenized['attention_mask'] = torch.stack(
                    [text[ii]['attention_mask'] for ii in range(len(text))],
                    dim=0).squeeze(1).to(device)[:, :tokenized['attention_mask'].shape[-1]]
                encoded_text = self.text_encoder(**tokenized)

                # Transpose memory because pytorch's attention expects sequence first
                text_memory = encoded_text.last_hidden_state.transpose(0, 1)
                # Invert attention mask that we get from huggingface because its the opposite in pytorch transformer
                text_attention_mask = tokenized.attention_mask.ne(1).bool()

                # Resize the encoder hidden states to be of the same d_model as the decoder
                ## (#words, B, 768), (#words, B, 256)
                text_memory_resized = self.resizer(text_memory)

            ##############################################
            # Concat on the sequence dimension
            src = torch.cat([self.norm(src), text_memory_resized], dim=0)
            # For mask, sequence dimension is second
            mask = torch.cat([mask, text_attention_mask], dim=1)
            # Pad the pos_embed with 0 so that the addition will be a no-op for the text tokens
            pos_embed = torch.cat([pos_embed, torch.zeros_like(text_memory_resized)], dim=0)
            ###############################################

            img_memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

            text_memory = img_memory[-len(text_memory_resized):]

            assert img_memory.shape[1] == text_memory.shape[1] == tgt.shape[1]
            memory_cache = {
                "text_memory_resized": text_memory_resized,
                "text_memory": text_memory,
                "img_memory": img_memory,
                "text_pooled_op": None,
                "img_pooled_op": None,
                "mask": mask,
                "text_attention_mask": text_attention_mask,
                "pos_embed": pos_embed,
                "query_embed": query_embed,
                "tokenized": tokenized,
            }
            return memory_cache

        else:
            ## updated with a separate class
            tgt = self.embedding(prev_indx).permute(1, 0, 2)
            query_embed = self.embedding.position_embeddings.weight.unsqueeze(1)
            query_embed = query_embed.repeat(1, tgt.shape[1], 1)

            assert img_memory.shape[1] == text_memory.shape[1] == tgt.shape[1]
            ## tgt_mask defined later in decoder_layer
            hs = self.decoder(
                tgt,
                img_memory,
                text_memory,
                memory_key_padding_mask=mask,
                text_memory_key_padding_mask=text_attention_mask,
                pos=pos_embed,
                query_pos=query_embed[:len(tgt)],
            )
            return hs.transpose(1, 2)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):

        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt,
        memory,
        text_memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        text_memory_key_padding_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(
                output,
                memory,
                text_memory=text_memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                text_memory_key_padding_mask=text_memory_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )

            if self.return_intermediate:
                intermediate.append(output)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class MultiheadAttention(nn.Module):

    def __init__(self,
                 embed_dim,
                 num_heads,
                 dropout=0.,
                 bias=True,
                 batch_first=False,
                 cross_attn=False,
                 device=None,
                 dtype=None) -> None:
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.cross_attn = cross_attn
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        if not cross_attn:
            self.attention_func = FlashAttention(softmax_scale=math.sqrt(self.head_dim), attention_dropout=self.dropout)
        else:
            self.attention_func = FlashCrossAttention(softmax_scale=math.sqrt(self.head_dim), attention_dropout=self.dropout)
        self.in_proj_weight = nn.Parameter(torch.empty((3, embed_dim, embed_dim), **factory_kwargs))
        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3, embed_dim, **factory_kwargs))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor,
                key_padding_mask: Optional[Tensor] = None,
                causal: bool = False) -> Tensor:
        if not self.cross_attn:
            qkv = torch.stack((query, key, value), dim=1)
            qkv = torch.matmul(qkv, self.in_proj_weight).transpose(1, 2)
            if self.in_proj_bias is not None:
                qkv = qkv + self.in_proj_bias
            qkv = rearrange(qkv, 's b three (h d) -> b s three h d', three=3, h=self.num_heads)
            context, _ = self.attention_func(qkv.half(), key_padding_mask=key_padding_mask, causal=causal)
        else:
            q = torch.matmul(query, self.in_proj_weight[0])  # [s,b,h*d]
            kv = torch.stack((key, value), dim=1)
            kv = torch.matmul(kv, self.in_proj_weight[1:]).transpose(1, 2)  # [s,b,2,h*d]
            if self.in_proj_bias is not None:
                q = q + self.in_proj_bias[0]
                kv = kv + self.in_proj_bias[1:]
            q = rearrange(q, 's b (h d) -> b s h d', h=self.num_heads)
            kv = rearrange(kv, 's b two (h d) -> b s two h d', two=2, h=self.num_heads)
            context = self.attention_func(q.half(), kv.half())

        context = rearrange(context, 'b s h d -> s b (h d)')
        out = self.out_proj(context)
        return out.float()


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = FusedLayerNorm(d_model)
        self.norm2 = FusedLayerNorm(d_model)
        # self.dropout1 = nn.Dropout(dropout)
        # self.dropout2 = nn.Dropout(dropout)
        self.dropout1 = dropout

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, key_padding_mask=src_key_padding_mask)
        # src = src + self.dropout1(src2)
        src = dropout_add(src2, src, self.dropout1, self.training)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # src = src + self.dropout2(src2)
        src = dropout_add(src2, src, self.dropout1, self.training)
        src = self.norm2(src)
        return src

    def forward_pre(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, key_padding_mask=src_key_padding_mask)
        # src = src + self.dropout1(src2)
        src = dropout_add(src2, src, self.dropout, self.training)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        # src = src + self.dropout2(src2)
        src = dropout_add(src2, src, self.dropout, self.training)
        return src

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn_image = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.cross_attn_text = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = FusedLayerNorm(d_model)
        # self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = FusedLayerNorm(d_model)
        self.norm4 = FusedLayerNorm(d_model)
        # self.dropout1 = nn.Dropout(dropout)
        # # self.dropout2 = nn.Dropout(dropout)
        # self.dropout3 = nn.Dropout(dropout)
        # self.dropout4 = nn.Dropout(dropout)
        self.dropout1 = dropout

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    # For now, trying one version where its self attn -> cross attn text -> cross attn image -> FFN
    def forward_post(
        self,
        tgt,
        memory,
        text_memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        text_memory_key_padding_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(tgt, query_pos)

        # tgt_mask = ~(torch.tril(torch.ones((tgt.shape[0], tgt.shape[0]))).to(tgt.device).bool())

        # Self attention
        tgt2 = self.self_attn(q, k, value=tgt, key_padding_mask=tgt_key_padding_mask, causal=True)
        # tgt = tgt + self.dropout1(tgt2)
        tgt = dropout_add(tgt2, tgt, self.dropout1, self.training)
        tgt = self.norm1(tgt)

        # Cross attention to image
        tgt2 = self.cross_attn_image(query=self.with_pos_embed(tgt, query_pos),
                                     key=self.with_pos_embed(memory, pos),
                                     value=memory,
                                     key_padding_mask=memory_key_padding_mask)[0]
        # tgt = tgt + self.dropout3(tgt2)
        tgt = dropout_add(tgt2, tgt, self.dropout1, self.training)
        tgt = self.norm3(tgt)

        # FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        # tgt = tgt + self.dropout4(tgt2)
        tgt = dropout_add(tgt2, tgt, self.dropout1, self.training)
        tgt = self.norm4(tgt)
        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        text_memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        text_memory_key_padding_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        assert False, "not implemented yet"
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        text_memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        text_memory_key_padding_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, pos,
                                    query_pos)
        return self.forward_post(
            tgt,
            memory,
            text_memory,
            tgt_mask,
            memory_mask,
            text_memory_key_padding_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
        )


class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = FusedLayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        pass_pos_and_query=True,
        text_encoder_type=args.text_encoder_type,
        freeze_text_encoder=args.freeze_text_encoder,
        num_queries=args.num_queries,
        max_decoding_step=args.max_decoding_step,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
