"""Bucle de generacion transparente: las mismas operaciones que generate() de Microsoft, en el mismo orden,
pero con la cache y el ruido a la vista. Es la base de todo lo que mira o toca la red por dentro:

  - enganches por fotograma: `al_condicion(cond, neg, j)` (dirigir la condicion, E2), `ganancia_texto`
    (enfasis por ficha, §5.3 del plan de la red), `neg_tts_lm` propio (prefijo negativo, §5.1);
  - registro por fotograma: condicion positiva y negativa, residual de cada capa en la posicion que condiciona,
    latente, probabilidad de fin, ventana de texto leida;
  - `foto()` / `reponer()`: el estado entero (4 caches KV, colas del decodificador, contadores, RNG) para
    bifurcar una locucion desde un fotograma (§5.4).

PARIDAD: con la misma semilla, `Generador(...).correr()` da el mismo audio que generate() muestra a muestra
(probar_mecanica.py lo comprueba con pesos aleatorios; con los reales vale igual porque no depende de los pesos).
"""
import copy

import torch

from modelo import IMAGE_PAD

VENTANA_TEXTO, VENTANA_VOZ = 5, 6


def _cache_viva(c):
    """La cache del .pt viene como DynamicCache antigua (key_cache/value_cache); transformers >= 4.57 pide .layers."""
    from vibevoice.modular.modeling_vibevoice_streaming_inference import _ensure_cache_has_layers
    return _ensure_cache_has_layers(c)


class Generador:
    def __init__(self, modelo, base, ids_texto, cfg_scale=3.0, pasos=None, al_condicion=None,
                 ganancia_texto=None, neg_tts_lm=None, registrar=False, residuales=False, atenciones=False):
        self.m = modelo
        self.cfg = cfg_scale
        self.al_condicion = al_condicion
        self.ganancia_texto = ganancia_texto or {}       # {indice de ficha del texto: factor}
        self.registrar, self.residuales, self.atenciones = registrar, residuales, atenciones
        if pasos:
            modelo.set_ddpm_inference_steps(pasos)
        base = copy.deepcopy(base)
        if neg_tts_lm is not None:
            base["neg_tts_lm"] = neg_tts_lm
        self.ids = torch.tensor([ids_texto], dtype=torch.long)
        self.c_lm, self.c_tts = _cache_viva(base["lm"].past_key_values), _cache_viva(base["tts_lm"].past_key_values)
        self.c_neg_lm, self.c_neg = _cache_viva(base["neg_lm"].past_key_values), _cache_viva(base["neg_tts_lm"].past_key_values)
        self.h_tts = base["tts_lm"].last_hidden_state.float()
        self.h_neg = base["neg_tts_lm"].last_hidden_state.float()
        self.L_lm, self.L_tts, self.L_neg = self.h_lm_len(base), self.h_tts.shape[1], self.h_neg.shape[1]
        from vibevoice.modular.modular_vibevoice_tokenizer import VibeVoiceTokenizerStreamingCache
        self.acustica = VibeVoiceTokenizerStreamingCache()
        self.ventana = 0
        self.fotograma = 0
        self._en_negativo = False
        self.audio, self.reg = [], []
        self._res = []
        if residuales:
            for capa in self.m.model.tts_language_model.layers:
                capa.register_forward_hook(self._guardar_residual)

    @staticmethod
    def h_lm_len(base):
        return base["lm"].last_hidden_state.shape[1]

    # ---- estado ----
    @torch.no_grad()
    def foto(self):
        return dict(c_lm=copy.deepcopy(self.c_lm), c_tts=copy.deepcopy(self.c_tts), c_neg_lm=copy.deepcopy(self.c_neg_lm),
                    c_neg=copy.deepcopy(self.c_neg), h_tts=self.h_tts.clone(), h_neg=self.h_neg.clone(),
                    L=(self.L_lm, self.L_tts, self.L_neg), acustica=copy.deepcopy(self.acustica),
                    ventana=self.ventana, fotograma=self.fotograma, rng=torch.get_rng_state(), n_audio=len(self.audio))

    def reponer(self, f):
        self.c_lm, self.c_tts, self.c_neg_lm, self.c_neg = (copy.deepcopy(f[k]) for k in ("c_lm", "c_tts", "c_neg_lm", "c_neg"))
        self.h_tts, self.h_neg = f["h_tts"].clone(), f["h_neg"].clone()
        self.L_lm, self.L_tts, self.L_neg = f["L"]
        self.acustica = copy.deepcopy(f["acustica"])
        self.ventana, self.fotograma = f["ventana"], f["fotograma"]
        torch.set_rng_state(f["rng"])
        self.audio = self.audio[: f["n_audio"]]
        self.reg = self.reg[: f["fotograma"]]

    # ---- pasadas ----
    def _guardar_residual(self, mod, ent, sal):
        h = sal[0] if isinstance(sal, tuple) else sal
        self._res.append(h[0, -1].detach().clone())

    def _lm(self, ids):
        S = ids.shape[1]
        pos = torch.arange(self.L_lm, self.L_lm + S)[None]
        out = self.m.forward_lm(input_ids=ids, attention_mask=torch.ones(1, self.L_lm + S, dtype=torch.long),
                                position_ids=pos, cache_position=pos[0], past_key_values=self.c_lm, use_cache=True, return_dict=True)
        self.c_lm = out.past_key_values
        self.L_lm += S
        return out.last_hidden_state

    def _tts(self, emb, tipo_texto, negativo=False):
        S = emb.shape[1]
        L = self.L_neg if negativo else self.L_tts
        pos = torch.arange(L, L + S)[None]
        self._res = []
        self._en_negativo = negativo
        out = self.m.forward_tts_lm(input_ids=torch.full((1, S), IMAGE_PAD, dtype=torch.long),
                                    attention_mask=torch.ones(1, L + S, dtype=torch.long), position_ids=pos, cache_position=pos[0],
                                    past_key_values=self.c_neg if negativo else self.c_tts, use_cache=True, return_dict=True,
                                    lm_last_hidden_state=emb, tts_text_masks=torch.full((1, 1), int(tipo_texto), dtype=torch.long),
                                    output_attentions=self.atenciones and not negativo)
        if negativo:
            self.c_neg, self.L_neg, self.h_neg = out.past_key_values, L + S, out.last_hidden_state
        else:
            self.c_tts, self.L_tts, self.h_tts = out.past_key_values, L + S, out.last_hidden_state
            self._ultimas_res = list(self._res)
            self._ultimas_att = out.attentions
        return out

    def _texto(self):
        a, b = self.ventana * VENTANA_TEXTO, (self.ventana + 1) * VENTANA_TEXTO
        ids = self.ids[:, a:b]
        self.ventana += 1
        if ids.shape[1] == 0:
            return False
        h = self._lm(ids)
        if self.ganancia_texto:
            h = h.clone()
            for k, g in self.ganancia_texto.items():
                if a <= k < b:
                    h[0, k - a] = h[0, k - a] * g
        self._tts(h, 1)
        return True

    def _voz(self):
        m = self.m.model
        cond, neg = self.h_tts[:, -1, :], self.h_neg[:, -1, :]
        if self.al_condicion is not None:
            cond, neg = self.al_condicion(cond, neg, self.fotograma)
        lat = self.m.sample_speech_tokens(cond, neg, cfg_scale=self.cfg).unsqueeze(1)
        z = lat / m.speech_scaling_factor - m.speech_bias_factor
        trozo = m.acoustic_tokenizer.decode(z, cache=self.acustica, sample_indices=torch.LongTensor([0]), use_cache=True)
        self.audio.append(trozo[0].detach().clone())
        if self.registrar:
            self.reg.append(dict(cond=cond[0].detach().clone(), neg=neg[0].detach().clone(), lat=lat[0, 0].detach().clone(),
                                 ventana=self.ventana - 1,
                                 res=torch.stack(self._ultimas_res) if self.residuales and self._ultimas_res else None,
                                 att=self._resumen_atencion() if self.atenciones else None))
        emb = m.acoustic_connector(lat)
        out = self._tts(emb, 0)
        self._tts(emb, 0, negativo=True)
        self.fotograma += 1
        fin = torch.sigmoid(out.logits[0]).item() > 0.5
        if self.registrar:
            self.reg[-1]["p_fin"] = torch.sigmoid(out.logits[0]).item()
        return fin

    def _resumen_atencion(self):
        """Masa de atencion de cada cabeza (capa x cabeza) sobre [latentes del prefijo, texto del prefijo, lo generado]."""
        if not self._ultimas_att:
            return None
        N = self.L_tts_prefijo_latentes
        M = self.L_tts_prefijo - N
        filas = []
        for a in self._ultimas_att:            # [1, cabezas, 1, L]
            w = a[0, :, -1, :]
            filas.append(torch.stack([w[:, :N].sum(-1), w[:, N:N + M].sum(-1), w[:, N + M:].sum(-1)], -1))
        return torch.stack(filas)              # [capas, cabezas, 3]

    @torch.no_grad()
    def correr(self, max_fotogramas=600, parar_en=None):
        """Genera hasta el fin (o `parar_en` fotogramas). Devuelve la onda [muestras]."""
        self.L_tts_prefijo = self.L_tts
        self.L_tts_prefijo_latentes = self.L_tts - self.L_lm
        while self.fotograma < max_fotogramas:
            if parar_en is not None and self.fotograma >= parar_en:
                break
            self._texto()
            fin = False
            for _ in range(VENTANA_VOZ):
                fin = self._voz()
                if fin or (parar_en is not None and self.fotograma >= parar_en):
                    break
            if fin:
                break
        return self.onda()

    def onda(self):
        return torch.cat(self.audio, dim=-1).flatten() if self.audio else torch.zeros(0)
