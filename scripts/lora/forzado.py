"""Pasada FORZADA (teacher forcing) de VibeVoice-Realtime-0.5B, diferenciable: el nucleo del bucle de
entrenamiento del plan de mejora (F7). Microsoft no publico el bucle de este modelo; esto reproduce, en
UNA pasada por el transformador, exactamente la secuencia que arma generate() fotograma a fotograma:

  prefijo de voz  (receta de clonar_voz.py, docs/clonado-de-voz.md §3):
     lm      : el texto de la referencia
     tts_lm  : [conector(latentes de la referencia) (N), estados del lm sobre ese texto (M)], todo tipo 0
  objetivo (como generate):
     lm      : continua con el texto a decir, de corrido
     tts_lm  : ventana de 5 fichas de texto (tipo 1, estados del lm) -> 6 fotogramas de voz (tipo 0,
               conector del latente) -> siguiente ventana ... ; agotado el texto, solo voz
  la condicion del fotograma j es el estado del tts_lm en la posicion ANTERIOR a donde entra su latente;
  el clasificador de fin mira el estado en la posicion del latente j (1 en el ultimo)
  rama negativa (guia sin clasificador): [<|image_pad|> por lm y tts_lm, tipo 1] + los mismos latentes

Perdidas:
  difusion : la cabeza predice la velocidad (v-prediction, programa coseno de 1000 pasos) del latente real
             (escalado: (z + sesgo) * escala, lo que ve la cabeza al generar) con ruido a un t al azar
  fin      : entropia cruzada binaria del clasificador de fin
  negativa : la misma perdida de difusion con la condicion de la rama negativa, en una fraccion de los
             fotogramas: el LoRA toca el tts_lm, que tambien calcula esa rama, y la guia no debe romperse
"""
import math

import torch
import torch.nn.functional as F

VENTANA_TEXTO = 5
VENTANA_VOZ = 6


def disposicion(n_texto, n_voz):
    """Orden de las posiciones del objetivo en el tts_lm: lista de ('t', k) y ('v', j)."""
    orden, k, j = [], 0, 0
    while k < n_texto or j < n_voz:
        if k < n_texto:
            w = min(VENTANA_TEXTO, n_texto - k)
            orden += [("t", k + a) for a in range(w)]
            k += w
            cupo = VENTANA_VOZ
        else:
            cupo = n_voz - j
        for _ in range(min(cupo, n_voz - j)):
            orden.append(("v", j))
            j += 1
        if j >= n_voz and k < n_texto:
            return None            # el audio se acaba antes que el texto: ejemplo raro, se descarta
    return orden


class Programa:
    """El programa de ruido de la cabeza (coseno, 1000 pasos, v-prediction), para entrenar."""

    def __init__(self, n=1000, dispositivo="cpu"):
        s = 0.008
        t = torch.linspace(0, n, n + 1, dtype=torch.float64) / n
        f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        betas = torch.clamp(1 - f[1:] / f[:-1], 0, 0.999)
        self.alfa_bar = torch.cumprod(1 - betas, 0).float().to(dispositivo)
        self.n = n

    def ruido(self, x0, t, eps):
        a = self.alfa_bar[t].sqrt()[:, None]
        b = (1 - self.alfa_bar[t]).sqrt()[:, None]
        return a * x0 + b * eps, a * eps - b * x0            # (x_t, velocidad objetivo)


def latentes(modelo, x, dispositivo, muestrear=True):
    """Onda (24 kHz, mono, float) -> latentes escalados como los ve la cabeza [T, 64]."""
    m = modelo.model
    muestras = 3200                                              # 24000 / 7,5
    n = math.ceil(len(x) / muestras)
    with torch.no_grad():
        e = m.acoustic_tokenizer.encode(torch.as_tensor(x, dtype=torch.float32)[None, None].to(dispositivo))
        z = e.sample(dist_type=m.acoustic_tokenizer.std_dist_type)[0] if muestrear else e.mean
        return ((z + m.speech_bias_factor) * m.speech_scaling_factor)[0, :n].float()


def pasada(modelo, tok, ej, programa, frac_negativa=0.15, peso_fin=0.1):
    """Una pasada forzada. ej: {'ref_lat': [N,64], 'ref_txt': str, 'lat': [T,64], 'txt': str}.
    Devuelve (perdida total, desglose)."""
    m = modelo.model
    d = next(modelo.parameters()).device
    ref_ids = tok.encode(ej["ref_txt"], add_special_tokens=False)
    txt_ids = tok.encode(ej["txt"].strip() + "\n", add_special_tokens=False)
    lat_ref, lat = ej["ref_lat"].to(d), ej["lat"].to(d)
    orden = disposicion(len(txt_ids), lat.shape[0])
    if orden is None:
        return None, None
    # 1. lm sobre [texto de referencia ; texto a decir], de corrido (causal, igual que su cache)
    ids = torch.tensor([ref_ids + txt_ids], device=d)
    h_lm = modelo.forward_lm(input_ids=ids, attention_mask=torch.ones_like(ids), return_dict=True).last_hidden_state[0]
    M = len(ref_ids)
    # 2. tts_lm: prefijo (tipo 0) + objetivo intercalado
    con_ref = m.acoustic_connector(lat_ref[None])[0]
    con = m.acoustic_connector(lat[None])[0]
    filas, tipos, pos_voz = [con_ref, h_lm[:M]], [0] * (lat_ref.shape[0] + M), {}
    for tipo, i in orden:
        if tipo == "t":
            filas.append(h_lm[M + i][None])
            tipos.append(1)
        else:
            pos_voz[i] = len(tipos)
            filas.append(con[i][None])
            tipos.append(0)
    emb = torch.cat(filas, 0)[None]
    emb = emb + m.tts_input_types(torch.tensor([tipos], device=d))
    h = m.tts_language_model(inputs_embeds=emb, attention_mask=torch.ones(1, emb.shape[1], device=d),
                             return_dict=True).last_hidden_state[0]
    T = lat.shape[0]
    idx = torch.tensor([pos_voz[j] for j in range(T)], device=d)
    cond = h[idx - 1]                                     # condicion del fotograma j: posicion anterior
    # 3. rama negativa: <|image_pad|> (tipo 1) + los latentes (tipo 0)
    neg_id = torch.tensor([[tok.convert_tokens_to_ids("<|image_pad|>")]], device=d)
    h_neg_lm = modelo.forward_lm(input_ids=neg_id, attention_mask=torch.ones_like(neg_id), return_dict=True).last_hidden_state
    emb_n = torch.cat([h_neg_lm[0], con], 0)[None]
    emb_n = emb_n + m.tts_input_types(torch.tensor([[1] + [0] * T], device=d))
    h_n = m.tts_language_model(inputs_embeds=emb_n, attention_mask=torch.ones(1, T + 1, device=d),
                               return_dict=True).last_hidden_state[0]
    cond_neg = h_n[:T]                                    # posicion anterior a cada latente
    # 4. difusion (positiva en todos los fotogramas, negativa en una fraccion)
    t = torch.randint(0, programa.n, (T,), device=d)
    eps = torch.randn_like(lat)
    xt, v = programa.ruido(lat, t, eps)
    pred = m.prediction_head(xt, t.float(), condition=cond)
    l_dif = F.mse_loss(pred, v)
    sel = torch.rand(T, device=d) < frac_negativa
    l_neg = torch.zeros((), device=d)
    if sel.any():
        pred_n = m.prediction_head(xt[sel], t[sel].float(), condition=cond_neg[sel])
        l_neg = F.mse_loss(pred_n, v[sel])
    # 5. fin: el estado en la posicion de cada latente decide si se para despues de el
    logit = modelo.tts_eos_classifier(h[idx]).squeeze(-1)
    fin = torch.zeros(T, device=d)
    fin[-1] = 1.0
    l_fin = F.binary_cross_entropy_with_logits(logit, fin, pos_weight=torch.tensor(float(T), device=d))
    total = l_dif + frac_negativa * l_neg + peso_fin * l_fin
    return total, {"dif": float(l_dif), "neg": float(l_neg), "fin": float(l_fin), "T": T,
                   "cond": cond.detach(), "cond_neg": cond_neg.detach()}
