"""La rama negativa del CFG como mando de estilo (plan de la red, §5.1).

Hoy la rama negativa arranca de <|image_pad|> y solo ve lo que se va generando. Aqui se fabrica un `neg_tts_lm`
que ademas lleva DELANTE unos latentes propios (la misma persona hablando plano, triste, sin pausas...), con la
receta del prefijo (docs/clonado-de-voz.md §3, pasos 3 y 4): [h(<|image_pad|>) tipo 1] + [conector(z) tipo 0].
La guia `neg + cfg (cond - neg)` empuja entonces lejos de ESE registro. Cero coste en tiempo de ejecucion.

    neg = construir(modelo, tok, latentes)        # latentes [N, 64] ya escalados como los ve la cabeza
    Generador(modelo, base, ids, neg_tts_lm=neg)  # o guardarlo en el .pt de la voz como "neg_tts_lm"

Latentes escalados: los de `Generador(registrar=True)` (lo generado) o `forzado.latentes()` (audio real por el
codificador comunitario).
"""
import torch
from transformers.modeling_outputs import BaseModelOutputWithPast

from modelo import IMAGE_PAD


@torch.no_grad()
def construir(modelo, tok, latentes):
    m = modelo.model
    d = next(modelo.parameters()).device
    neg_id = torch.tensor([[IMAGE_PAD]], device=d)
    h_neg = modelo.forward_lm(input_ids=neg_id, attention_mask=torch.ones_like(neg_id), return_dict=True).last_hidden_state
    con = m.acoustic_connector(latentes.to(d)[None].to(h_neg.dtype))
    emb = torch.cat([h_neg, con], 1)
    tipos = torch.tensor([[1] + [0] * latentes.shape[0]], device=d)
    emb = emb + m.tts_input_types(tipos)
    out = m.tts_language_model(inputs_embeds=emb, attention_mask=torch.ones(1, emb.shape[1], dtype=torch.long, device=d),
                               use_cache=True, return_dict=True)
    return BaseModelOutputWithPast(last_hidden_state=out.last_hidden_state, past_key_values=out.past_key_values)
