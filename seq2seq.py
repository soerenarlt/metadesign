"""
Transformer model for sequence to sequence translation. Based on nanogpt repo.
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class Attention(nn.Module):

    def __init__(self, config, causal=False, self_attention=False):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.causal = causal
        self.self_attention = self_attention
        # key, query, value projections for all heads, but in a batch
        if self.self_attention:
            self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        else:
            self.c_attn1= nn.Linear(config.n_embd, 2 * config.n_embd, bias=config.bias)
            self.c_attn2= nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
            
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention only supported in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.tgt_len, config.tgt_len))
                                        .view(1, 1, config.tgt_len, config.tgt_len))

    def forward(self, x, encoder_out=None):
        if self.self_attention:
            B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
            
            # calculate query, key, values for all heads in batch and move head forward to be the batch dim
            q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
            k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        else: # cross-attention, take in encoder output as keys and values
            assert encoder_out is not None
            B, T_enc, C = encoder_out.size() # batch size, sequence length, embedding dimensionality (n_embd)

            # calculate query, key, values for all heads in batch and move head forward to be the batch dim
            k, v  = self.c_attn1(encoder_out).split(self.n_embd, dim=2)
            k = k.view(B, T_enc, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            v = v.view(B, T_enc, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

            # get query from decoder input
            T = x.size(1)
            q = self.c_attn2(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=self.causal)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if self.causal:
                att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
    
class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

@dataclass
class Seq2SeqConfig:
    src_len: int = 1024
    tgt_len: int = 1024
    src_vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    tgt_vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    src_pad_token_id: int = None
    tgt_pad_token_id: int = None

class EncoderBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = Attention(config, causal=False, self_attention=True)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    
class DecoderBlock(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn1 = Attention(config, causal=True, self_attention=True)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn2 = Attention(config, causal=False, self_attention=False)
        self.ln_3 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, encoder_out):
        x = x + self.attn1(self.ln_1(x))
        x = x + self.attn2(self.ln_2(x), encoder_out)
        x = x + self.mlp(self.ln_3(x))
        return x


class Seq2Seq(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.src_vocab_size is not None
        assert config.tgt_vocab_size is not None
        assert config.src_len is not None
        assert config.tgt_len is not None
        self.config = config

        self.encoder = nn.ModuleDict(dict(
            wte = nn.Embedding(config.src_vocab_size, config.n_embd),
            wpe = nn.Embedding(config.src_len, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([EncoderBlock(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))

        self.decoder = nn.ModuleDict(dict(
            wte = nn.Embedding(config.tgt_vocab_size, config.n_embd),
            wpe = nn.Embedding(config.tgt_len, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.tgt_vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.decoder.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.decoder.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, src, idx, targets=None):
        device = src.device
        b, t = src.size()
        assert t <= self.config.src_len, f"Cannot forward sequence of length {t}, block size is only {self.config.src_len}"
        pos_src = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the encoder block
        tok_emb = self.encoder.wte(src) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.encoder.wpe(pos_src) # position embeddings of shape (t, n_embd)
        x = self.encoder.drop(tok_emb + pos_emb)
        for block in self.encoder.h:
            x = block(x)
        encoder_out = self.encoder.ln_f(x)

        b, t = idx.size()
        assert t <= self.config.tgt_len, f"Cannot forward sequence of length {t}, block size is only {self.config.tgt_len}"
        pos_tgt = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the decoder block
        tok_emb = self.decoder.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.decoder.wpe(pos_tgt) # position embeddings of shape (t, n_embd)
        x = self.decoder.drop(tok_emb + pos_emb)
        for block in self.decoder.h:
            x = block(x, encoder_out)
        x = self.decoder.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=self.config.tgt_pad_token_id)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss 
    

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        # print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        # print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        use_fused = False #disable fused for now, it seems to be causing problems
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9,0.95), **extra_args)
        # print(f"using fused AdamW: {use_fused}")

        return optimizer
    

    @torch.no_grad()    
    def generate(self, src, start_token_id=None, end_token_id=None, top_k=None, top_p=None, temperature=1.0):
        """
        Generate a sequence from a conditioning sequence of indices using both top-k and top-p sampling.
        """
        device = src.device
        idx = torch.full((1, 1), start_token_id, dtype=torch.long, device=device)
        for _ in range(self.config.tgt_len - 1):
            # Forward the model to get the logits
            logits, _ = self(src, idx)
            # Scale logits by the temperature
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering if specified
            if top_k is not None:
                values, indices = torch.topk(logits, top_k)
                logits[logits < values[:, [-1]]] = -float('Inf')

            # Apply top-p filtering if specified
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = -float('Inf')

            # Convert logits to probabilities and sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
            # Check for end token
            if end_token_id is not None and (idx_next == end_token_id).all():
                break
            # Check if we reach target length
            if idx.size(1) >= self.config.tgt_len:
                break

        return idx