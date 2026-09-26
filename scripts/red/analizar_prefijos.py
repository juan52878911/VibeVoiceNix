#!/usr/bin/env python3
"""Los prefijos de voz (.pt) como estados internos REALES del 0.5B, sin necesitar los pesos.

    python3 scripts/red/analizar_prefijos.py <carpeta con los .pt> [salida.json]

Cada .pt guarda la salida del tts_lm en todas las posiciones del prefijo (el flujo de la condicion) y la cache
K/V de sus 20 capas: es lo que el modelo "piensa" de una voz real. Mide donde se separan las voces y el sexo
por capa (fraccion de varianza entre voces; sexo con deja-una-voz-fuera), normas, PCA y vecinos. Resultado del
26-09 en docs/bancos/2026-09-26-red-interna-entorno.md.
tts_lm.last_hidden_state [T, 896]: T = N latentes (primero) + M texto. Es el flujo de la condicion.
past_key_values: 20 capas de K (con RoPE) y V (sin RoPE), 2 cabezas KV x 64.
"""
import sys
import torch, glob, os, json, numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.decomposition import PCA
torch.manual_seed(0); np.random.seed(0)
voces = {}
for f in sorted(glob.glob(os.path.join(os.path.expanduser(sys.argv[1]), '*.pt'))):
    d = torch.load(f, map_location='cpu', weights_only=False)
    n = os.path.basename(f)[:-3]
    M = d['lm'].last_hidden_state.shape[1]
    h = d['tts_lm'].last_hidden_state[0].float()          # [T, 896]
    V = torch.stack([v[0] for v in d['tts_lm'].past_key_values.value_cache]).float()  # [20, 2, T, 64]
    K = torch.stack([k[0] for k in d['tts_lm'].past_key_values.key_cache]).float()
    T = h.shape[0]; N = T - M
    voces[n] = dict(M=M, N=N, h_lat=h[:N], h_txt=h[N:], V_lat=V[:, :, :N].permute(2, 0, 1, 3).reshape(N, 20, 128),
                    K_lat=K[:, :, :N].permute(2, 0, 1, 3).reshape(N, 20, 128), sexo=int('woman' in n), idioma=n.split('-')[0])
nombres = list(voces)
res = {}
# 1. normas
rms_lat = np.mean([voces[n]['h_lat'].pow(2).mean(-1).sqrt().mean().item() for n in nombres])
rms_txt = np.mean([voces[n]['h_txt'].pow(2).mean(-1).sqrt().mean().item() for n in nombres])
res['rms_condicion'] = dict(latentes=round(rms_lat, 2), texto=round(rms_txt, 2))
# frames submuestreados para no sesgar por longitud
def muestras(clave, k=100):
    X, y_voz, y_sexo, y_idi = [], [], [], []
    for i, n in enumerate(nombres):
        x = voces[n][clave]; idx = torch.linspace(3, len(x) - 4, k).long()   # evita bordes
        X.append(x[idx].reshape(k, -1)); y_voz += [i] * k; y_sexo += [voces[n]['sexo']] * k; y_idi += [voces[n]['idioma']] * k
    return torch.cat(X).numpy(), np.array(y_voz), np.array(y_sexo), np.array(y_idi)
def fisher(X, y):
    mu = X.mean(0); clases = np.unique(y)
    entre = sum((y == c).sum() * ((X[y == c].mean(0) - mu) ** 2).sum() for c in clases)
    dentro = sum(((X[y == c] - X[y == c].mean(0)) ** 2).sum() for c in clases)
    return float(entre / (entre + dentro))
def lovo(X, y, grupos):
    """deja-una-voz-fuera: generaliza a una voz nunca vista?"""
    ac = []
    for g in np.unique(grupos):
        tr, te = grupos != g, grupos == g
        clf = RidgeClassifier(alpha=10.0).fit(X[tr], y[tr]); ac.append((clf.predict(X[te]) == y[te]).mean())
    return float(np.mean(ac))
X, yv, ys, yi = muestras('h_lat')
res['condicion_latentes'] = dict(fraccion_varianza_por_voz=round(fisher(X, yv), 3), sexo_deja_una_voz_fuera=round(lovo(X, ys, yv), 3))
Xt, yvt, yst, _ = muestras('h_txt', 40)
res['condicion_texto'] = dict(fraccion_varianza_por_voz=round(fisher(Xt, yvt), 3), sexo_deja_una_voz_fuera=round(lovo(Xt, yst, yvt), 3))
# 2. por capa, valores y claves en posiciones de latente
XV, _, _, _ = muestras('V_lat'); XK, _, _, _ = muestras('K_lat')
XV = XV.reshape(len(yv), 20, 128); XK = XK.reshape(len(yv), 20, 128)
por_capa = []
for c in range(20):
    por_capa.append(dict(capa=c, V_voz=round(fisher(XV[:, c], yv), 3), V_sexo_lovo=round(lovo(XV[:, c], ys, yv), 3),
                         K_voz=round(fisher(XK[:, c], yv), 3)))
res['por_capa'] = por_capa
# 3. PCA de la condicion en latentes: cuanta varianza en pocas componentes, y si la 1a separa sexo
pca = PCA(20).fit(X); Z = pca.transform(X)
res['pca'] = dict(var_10=round(float(pca.explained_variance_ratio_[:10].sum()), 3), var_20=round(float(pca.explained_variance_ratio_.sum()), 3),
                  corr_pc_sexo=[round(float(abs(np.corrcoef(Z[:, j], ys)[0, 1])), 2) for j in range(5)])
# 4. vecino mas cercano por media de voz
medias = {n: voces[n]['h_lat'].mean(0) for n in nombres}
vec = {}
for n in nombres:
    s = {m: torch.cosine_similarity(medias[n], medias[m], dim=0).item() for m in nombres if m != n}
    vec[n] = max(s, key=s.get)
res['vecino_mas_cercano'] = vec
res['mismo_sexo_en_vecino'] = round(np.mean([voces[n]['sexo'] == voces[vec[n]]['sexo'] for n in nombres]), 2)
res['mismo_idioma_en_vecino'] = round(np.mean([voces[n]['idioma'] == voces[vec[n]]['idioma'] for n in nombres]), 2)
# 5. deriva dentro del prefijo: la condicion en latentes se mueve con la posicion? (norma por tercios)
ter = []
for n in nombres:
    h = voces[n]['h_lat']; k = len(h) // 3
    ter.append([h[i * k:(i + 1) * k].norm(dim=-1).mean().item() for i in range(3)])
res['norma_por_tercio_del_prefijo'] = [round(float(v), 1) for v in np.mean(ter, 0)]
json.dump(res, open(sys.argv[2] if len(sys.argv) > 2 else 'prefijos.json', 'w'), indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in res.items() if k not in ('vecino_mas_cercano',)}, indent=1, ensure_ascii=False))
print('vecinos:', vec)
