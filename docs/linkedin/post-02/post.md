# Post 2 — La tabla de optimización

Formato: carrusel. Subir `carrusel.pdf` como documento (o las tres `slide-*.png` como imágenes).
Las imágenes se regeneran con `python3 docs/linkedin/post-02/generar.py`.

## Texto

```
5,39 → 0,885.

Es el RTF de VibeVoice, un modelo de voz de 0,5B, corriendo en un i7 de 35 W sin GPU.
Por debajo de 1, genera el audio más rápido de lo que tardas en escucharlo.

6,1× más rápido, con cinco cambios y la misma CPU:

→ Cuantización int8 dinámica en PyTorch: 1,96×
→ 6 pasos de difusión en vez de 20: 1,26×
→ Motor OpenVINO: 2,00×
→ Decodificador sin convolución traspuesta: 1,11×
→ Todo el bucle de difusión en un solo grafo: 1,06×

Lo que no sale en la tabla pesa lo mismo: cada idea se midió contra una meta escrita antes de medir.
La iGPU se quedó a un 1 % de la meta. Se descartó igual.

Todo medido y documentado, también lo que no funcionó 👇
github.com/juan52878911/VibeVoiceNix

#AI #MachineLearning #ModelOptimization #MLEngineering #NixOS #SelfHosted #OpenSource #EdgeAI
```

## Texto alternativo

1. Portada. El RTF de VibeVoice 0.5B pasa de 5,39, tachado, a 0,885. 6,1 veces más rápido en la misma CPU,
   sin GPU y sin nube. RTF menor que 1 significa más rápido que el tiempo real.
2. Tabla con barras de RTF en un i7-8700T. Punto de partida fp32 con 20 pasos: 5,39. Cuantización int8
   dinámica: 2,75, 1,96 veces. 6 pasos de difusión en vez de 20: 2,18, 1,26 veces. Motor OpenVINO: 1,09,
   2 veces. Decodificador sin convolución traspuesta: 0,98, 1,11 veces. Bucle de difusión en un solo grafo:
   0,885, 1,06 veces.
3. Lo que no entró en la tabla, cada idea medida contra una meta fijada antes: la iGPU del propio i7, −6,1 %
   cuando la meta pedía −7 %; el LM en int8 en vez de int4, 9 % más lento y no suena mejor; dos pasadas del
   LM en paralelo, −7 % cuando la meta pedía −20 %; enseñar a la VM su topología SMT, +0,6 %, ruido;
   torch.compile, 1,00 veces.

## De dónde sale cada cifra

- Pasos 0-3: `docs/optimizacion.md`, tabla «El viaje completo».
- 0,98: `docs/optimizacion.md` paso 7 (VM de producción, OpenVINO).
- 0,885: `docs/optimizacion.md` paso 8, mediana del banco A/B de 238 clips.
- Ganancias de los dos últimos: 1,09 → 0,98 y 0,941 → 0,885 (el propio banco del paso 8).
- Lo que no entró: `docs/plan-rendimiento.md` (fases 2-4, balance) y la tabla «NO funcionó» de `docs/optimizacion.md`.
