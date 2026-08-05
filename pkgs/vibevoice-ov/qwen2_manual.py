"""Qwen2 decoder reescrito a mano para exportar a OpenVINO.

Por que no trazar el Qwen2Model de transformers 4.57: su forward pasa por
DynamicCache, create_causal_mask y utilidades no trazables sin los parches de
optimum-intel (que ademas exporta con input_ids y logits, y VibeVoice necesita
inputs_embeds y hidden states). Este modulo son ~90 lineas de ops planas que
torch.jit.trace traga sin protestar, con la KV cache como entrada/salida
explicita (apilada en un tensor [capas, 1, kv_heads, T, 64]).

La equivalencia numerica con el Qwen2Model de HF se verifica en bench_lm.py.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

D = 896          # hidden
NH = 14          # cabezas de atencion
NKV = 2          # cabezas KV (GQA 7:1)
HD = 64          # dim por cabeza
DI = 4864        # intermedio MLP
THETA = 1e6      # rope


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rotar_mitad(x):
    x1, x2 = x[..., :HD // 2], x[..., HD // 2:]
    return torch.cat((-x2, x1), dim=-1)


class Capa(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = RMSNorm(D)
        self.post_attention_layernorm = RMSNorm(D)
        self.q_proj = nn.Linear(D, NH * HD, bias=True)
        self.k_proj = nn.Linear(D, NKV * HD, bias=True)
        self.v_proj = nn.Linear(D, NKV * HD, bias=True)
        self.o_proj = nn.Linear(NH * HD, D, bias=False)
        self.gate_proj = nn.Linear(D, DI, bias=False)
        self.up_proj = nn.Linear(D, DI, bias=False)
        self.down_proj = nn.Linear(DI, D, bias=False)

    def forward(self, h, cos, sin, mascara, pk, pv):
        B, S, _ = h.shape
        x = self.input_layernorm(h)
        q = self.q_proj(x).view(B, S, NH, HD).transpose(1, 2)
        k = self.k_proj(x).view(B, S, NKV, HD).transpose(1, 2)
        v = self.v_proj(x).view(B, S, NKV, HD).transpose(1, 2)
        # rope (convencion HF: cos/sin de dim completa, rotate_half)
        q = q * cos + rotar_mitad(q) * sin
        k = k * cos + rotar_mitad(k) * sin
        # concatenar cache: la salida ES la nueva cache
        k = torch.cat([pk, k], dim=2)
        v = torch.cat([pv, v], dim=2)
        # GQA sin materializar: enable_gqa deja que el kernel difunda 2->14
        # cabezas. Si el frontend de OV no lo soporta, poner GQA_MANUAL=1.
        import os
        if os.environ.get("GQA_MANUAL"):
            T = k.shape[2]
            k_e = k[:, :, None].expand(B, NKV, NH // NKV, T, HD).reshape(B, NH, T, HD)
            v_e = v[:, :, None].expand(B, NKV, NH // NKV, T, HD).reshape(B, NH, T, HD)
            a = F.scaled_dot_product_attention(q, k_e, v_e, attn_mask=mascara)
        else:
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=mascara, enable_gqa=True)
        a = a.transpose(1, 2).reshape(B, S, NH * HD)
        h = h + self.o_proj(a)
        x = self.post_attention_layernorm(h)
        h = h + self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return h, k, v


class Qwen2Manual(nn.Module):
    """Pila de capas + norm final. KV cache apilada como tensor unico."""

    def __init__(self, n_capas):
        super().__init__()
        self.capas = nn.ModuleList(Capa() for _ in range(n_capas))
        self.norm = RMSNorm(D)
        inv = 1.0 / (THETA ** (torch.arange(0, HD, 2, dtype=torch.float32) / HD))
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, inputs_embeds, position_ids, past_k, past_v):
        # inputs_embeds [1,S,896] f32 · position_ids [1,S] i64
        # past_k/past_v [n_capas,1,2,P,64] f32 (P puede ser 0)
        S = inputs_embeds.shape[1]
        P = past_k.shape[3]
        frec = position_ids[..., None].float() * self.inv_freq[None, None, :]
        emb = torch.cat([frec, frec], dim=-1)
        cos, sin = emb.cos()[:, None], emb.sin()[:, None]      # [1,1,S,64]
        # mascara causal general (vale para S=1 con pasado y para prefill)
        q_pos = P + torch.arange(S, device=inputs_embeds.device)
        k_pos = torch.arange(P + S, device=inputs_embeds.device)
        permitido = k_pos[None, :] <= q_pos[:, None]
        mascara = torch.where(permitido, 0.0, -1e9)[None, None].to(inputs_embeds.dtype)

        h = inputs_embeds
        ks, vs = [], []
        for i, capa in enumerate(self.capas):
            h, k, v = capa(h, cos, sin, mascara, past_k[i], past_v[i])
            ks.append(k)
            vs.append(v)
        return self.norm(h), torch.stack(ks), torch.stack(vs)


def cargar_desde_safetensors(ruta_st, prefijo, n_capas):
    """Carga solo `prefijo`.layers.* y `prefijo`.norm del safetensors (bf16->fp32)."""
    from safetensors import safe_open
    m = Qwen2Manual(n_capas)
    mapa = {"self_attn.q_proj": "q_proj", "self_attn.k_proj": "k_proj",
            "self_attn.v_proj": "v_proj", "self_attn.o_proj": "o_proj",
            "mlp.gate_proj": "gate_proj", "mlp.up_proj": "up_proj",
            "mlp.down_proj": "down_proj", "input_layernorm": "input_layernorm",
            "post_attention_layernorm": "post_attention_layernorm"}
    sd = {}
    with safe_open(ruta_st, framework="pt") as f:
        for k in f.keys():
            if not k.startswith(prefijo + "."):
                continue
            resto = k[len(prefijo) + 1:]
            if resto == "norm.weight":
                sd["norm.weight"] = f.get_tensor(k).float()
            elif resto.startswith("layers."):
                _, idx, *cola = resto.split(".")
                cuerpo = ".".join(cola[:-1])   # sin .weight/.bias
                suf = cola[-1]
                if cuerpo in mapa:
                    sd["capas.%s.%s.%s" % (idx, mapa[cuerpo], suf)] = f.get_tensor(k).float()
    faltan, sobran = m.load_state_dict(sd, strict=False)
    assert not faltan and not sobran, (faltan, sobran)
    m.eval()
    return m
