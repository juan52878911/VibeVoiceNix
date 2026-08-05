"""Decodificador acustico de VibeVoice reescrito con estado explicito.

Mismo truco que qwen2_manual: ops planas trazables, cada conv causal lleva su
estado (la cola de su entrada) como tensor de E/S, y luego make_stateful lo
convierte en variable interna de OpenVINO.

Con 1 frame latente por llamada TODAS las formas son estaticas:
  1 -> stem/etapa0 (2048c) -> x8 -> 8 (1024c) -> x5 -> 40 (512c) -> x5 -> 200
  (256c) -> x4 -> 800 (128c) -> x2 -> 1600 (64c) -> x2 -> 3200 (32c) -> 3200
  muestras de audio (24 kHz, un frame = 133 ms).

Estado cero == cache vacia del original: los SConv1d ya inicializan con ceros
y para SConvTranspose1d las contribuciones de entradas cero son nulas y el
recorte causal ('ultimos T*stride') selecciona exactamente las mismas
posiciones. La paridad lo confirma empiricamente.

Config fija del checkpoint: n_filters 32, ratios [8,5,5,4,2,2], depths
[8,3,3,3,3,3,3], mixer depthwise k7, RMSNorm eps 1e-5, gelu exacto,
layer_scale (gamma) activo, trim_right_ratio 1, pad 'constant'.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-5
RATIOS = [8, 5, 5, 4, 2, 2]
PROFUNDIDADES = [8, 3, 3, 3, 3, 3, 3]
DIMS = [2048, 1024, 512, 256, 128, 64, 32]   # dim de cada etapa tras su subida


def norma_rms_canal(x, peso):
    # ConvRMSNorm: normaliza el eje de canales; x [B,C,T]
    h = x.transpose(1, 2)
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + EPS)
    return (h * peso).transpose(1, 2)


class ConvCausal(nn.Module):
    """SConv1d k7 s1 en streaming: estado = ultimas 6 entradas."""

    def __init__(self, ent, sal, k=7):
        super().__init__()
        self.w = nn.Parameter(torch.empty(sal, ent, k))
        self.b = nn.Parameter(torch.empty(sal))
        self.ctx = k - 1

    def forward(self, x, est):
        full = torch.cat([est, x], dim=2)
        return F.conv1d(full, self.w, self.b), full[:, :, -self.ctx:]


class Bloque(nn.Module):
    """Block1D depthwise: RMSNorm -> dwconv(estado) -> gamma -> +res -> FFN."""

    def __init__(self, d):
        super().__init__()
        self.norm_w = nn.Parameter(torch.empty(d))
        self.dw_w = nn.Parameter(torch.empty(d, 1, 7))
        self.dw_b = nn.Parameter(torch.empty(d))
        self.gamma = nn.Parameter(torch.empty(d))
        self.ffn_norm_w = nn.Parameter(torch.empty(d))
        self.l1_w = nn.Parameter(torch.empty(4 * d, d))
        self.l1_b = nn.Parameter(torch.empty(4 * d))
        self.l2_w = nn.Parameter(torch.empty(d, 4 * d))
        self.l2_b = nn.Parameter(torch.empty(d))
        self.ffn_gamma_ = nn.Parameter(torch.empty(d))
        self.d = d

    def forward(self, x, est):
        r = x
        h = norma_rms_canal(x, self.norm_w)
        full = torch.cat([est, h], dim=2)
        nuevo = full[:, :, -6:]
        h = F.conv1d(full, self.dw_w, self.dw_b, groups=self.d)
        x = r + h * self.gamma[None, :, None]
        r = x
        h = norma_rms_canal(x, self.ffn_norm_w).transpose(1, 2)
        h = F.linear(h, self.l1_w, self.l1_b)
        h = F.gelu(h)
        h = F.linear(h, self.l2_w, self.l2_b).transpose(1, 2)
        return r + h * self.ffn_gamma_[None, :, None], nuevo


class SubidaTr(nn.Module):
    """SConvTranspose1d causal streaming: estado = ultimas k-1 ENTRADAS."""

    def __init__(self, ent, sal, ratio):
        super().__init__()
        k, s = ratio * 2, ratio
        self.w = nn.Parameter(torch.empty(ent, sal, k))
        self.b = nn.Parameter(torch.empty(sal))
        self.k, self.s = k, s

    def forward(self, x, est):
        full = torch.cat([est, x], dim=2)
        nuevo = full[:, :, -(self.k - 1):]
        y = F.conv_transpose1d(full, self.w, self.b, stride=self.s)
        y = y[:, :, : -(self.k - self.s)]          # recorte causal (todo a la dcha)
        return y[:, :, -(x.shape[2] * self.s):], nuevo   # solo lo nuevo


class Decodificador(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = ConvCausal(64, 2048)
        self.subidas = nn.ModuleList(
            SubidaTr(DIMS[i], DIMS[i + 1], RATIOS[i]) for i in range(6))
        self.etapas = nn.ModuleList(
            nn.ModuleList(Bloque(DIMS[i]) for _ in range(PROFUNDIDADES[i]))
            for i in range(7))
        self.head = ConvCausal(32, 1)
        # formas de los estados en orden de ejecucion
        self.formas_estado = [(64, 6)]                          # stem
        for i in range(7):
            if i > 0:
                self.formas_estado.append((DIMS[i - 1], RATIOS[i - 1] * 2 - 1))
            self.formas_estado += [(DIMS[i], 6)] * PROFUNDIDADES[i]
        self.formas_estado.append((32, 6))                      # head

    def forward(self, lat, *estados):
        nuevos = []
        j = 0

        def paso(mod, x, *extra):
            nonlocal j
            y, ne = mod(x, estados[j], *extra)
            nuevos.append(ne)
            j += 1
            return y

        x = paso(self.stem, lat)
        for i in range(7):
            if i > 0:
                x = paso(self.subidas[i - 1], x)
            for blq in self.etapas[i]:
                x = paso(blq, x)
        audio = paso(self.head, x)
        return (audio, *nuevos)


def cargar(ruta_st):
    from safetensors import safe_open
    m = Decodificador()
    P = "model.acoustic_tokenizer.decoder."
    with safe_open(ruta_st, framework="pt") as f:
        def t(k):
            return f.get_tensor(P + k).float()
        m.stem.w.data = t("upsample_layers.0.0.conv.conv.weight")
        m.stem.b.data = t("upsample_layers.0.0.conv.conv.bias")
        m.head.w.data = t("head.conv.conv.weight")
        m.head.b.data = t("head.conv.conv.bias")
        for i in range(6):
            m.subidas[i].w.data = t("upsample_layers.%d.0.convtr.convtr.weight" % (i + 1))
            m.subidas[i].b.data = t("upsample_layers.%d.0.convtr.convtr.bias" % (i + 1))
        for i in range(7):
            for jj, blq in enumerate(m.etapas[i]):
                b = "stages.%d.%d." % (i, jj)
                blq.norm_w.data = t(b + "norm.weight")
                blq.dw_w.data = t(b + "mixer.conv.conv.conv.weight")
                blq.dw_b.data = t(b + "mixer.conv.conv.conv.bias")
                blq.gamma.data = t(b + "gamma")
                blq.ffn_norm_w.data = t(b + "ffn_norm.weight")
                blq.l1_w.data = t(b + "ffn.linear1.weight")
                blq.l1_b.data = t(b + "ffn.linear1.bias")
                blq.l2_w.data = t(b + "ffn.linear2.weight")
                blq.l2_b.data = t(b + "ffn.linear2.bias")
                blq.ffn_gamma_.data = t(b + "ffn_gamma")
    m.eval()
    return m


def formas_estado_lista():
    """Formas [(canales, longitud)] de los 34 estados, sin instanciar pesos."""
    formas = [(64, 6)]
    for i in range(7):
        if i > 0:
            formas.append((DIMS[i - 1], RATIOS[i - 1] * 2 - 1))
        formas += [(DIMS[i], 6)] * PROFUNDIDADES[i]
    formas.append((32, 6))
    return formas


def estados_cero(m):
    return [torch.zeros(1, c, l) for c, l in m.formas_estado]
