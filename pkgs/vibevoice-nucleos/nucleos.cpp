// Nucleos nativos int8 (AVX2) para el decodificador acustico de VibeVoice.
//
// POR QUE EXISTE ESTE FICHERO
// La VM esta limitada por ancho de banda de memoria (medido: 80,7% del bus),
// asi que lo que paga es reducir los BYTES de peso que se leen por fotograma,
// no los FLOPs. quantize_dynamic de PyTorch solo toca nn.Linear: las
// convoluciones transpuestas del decodificador (40M de parametros, 162 MB
// leidos por fotograma en fp32) no las cuantiza nadie. Estos nucleos las
// sirven con pesos int8 (4x menos bytes) y de paso sustituyen tambien las
// FFN grandes para quitarse el sobrecoste por llamada de fbgemm con T
// pequeno (T = 1, 8 o 40 fotogramas segun la etapa).
//
// ESQUEMA DE CUANTIZACION
//   pesos:       int8 simetrico por canal de salida (escala fp32 por fila),
//                empaquetados UNA vez al cargar el modelo (nucleos_torch.py).
//   activaciones: dinamicas por tensor y por llamada, u7: [0,127] con punto
//                cero. u7 y no u8 A PROPOSITO: esta CPU (Coffee Lake) no
//                tiene VNNI y la ruta es VPMADDUBSW, que suma PARES u8*s8 en
//                i16 CON SATURACION. Con u8 un par puede valer 2*255*127 =
//                64770 > 32767 y satura en silencio; con u7 el maximo es
//                2*127*127 = 32258 < 32767: la saturacion es imposible por
//                construccion. Cuesta medio bit de activacion.
//   acumulacion: i32 (VPMADDWD contra unos), desescalado a fp32 al final.
//                Con K <= 8192: |acumulado| <= 2048 * 64516 ~ 1,3e8 << 2^31.
//
// La correccion del punto cero es cerrada: si x = s_x*(xq - zp) y
// w = s_w[n]*wq, entonces  y[t,n] = s_x*s_w[n] * (sum xq*wq - zp*sum wq).
// Las sumas de pesos (sum wq por fila) llegan precalculadas del empaquetado,
// para no releer los pesos una segunda vez en cada llamada.
//
// CONTRATO
//   - Todo fp32 de cara afuera; la cuantizacion de activaciones ocurre dentro.
//   - K (dimension de reduccion) debe ser multiplo de 32. Todas las capas que
//     se sustituyen lo cumplen (512..8192, y 448 en el stem); el envoltorio
//     Python se niega a sustituir cualquier capa que no.
//   - Un solo hilo llamante (el servicio serializa con un candado global);
//     dentro se paraleliza con OpenMP sobre canales de salida y se hereda
//     OMP_NUM_THREADS / OMP_PLACES del servicio.
//
// Compilacion (la hace nix/overlay.nix, aqui como referencia):
//   g++ -O3 -mavx2 -mfma -fopenmp -shared -fPIC -Wall nucleos.cpp -o libnucleos_vibevoice.so
// Solo AVX2+FMA: el i7-8700T no tiene AVX512 y el binario debe correr ahi.

#include <immintrin.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace {

// ---------------------------------------------------------------- cuantizar

// u7 asimetrica por tensor. El 0 entra siempre en el rango representable
// para que el punto cero sea exacto (las colas de silencio decodifican a 0).
struct EscalaAct {
    float escala;
    int32_t zp;
};

EscalaAct rango_u7(float mn, float mx) {
    if (mn > 0.f) mn = 0.f;
    if (mx < 0.f) mx = 0.f;
    float escala = (mx - mn) / 127.0f;
    if (!(escala > 0.f)) return {1.0f, 0};  // tensor todo ceros
    int32_t zp = (int32_t)nearbyintf(-mn / escala);
    if (zp < 0) zp = 0;
    if (zp > 127) zp = 127;
    return {escala, zp};
}

uint8_t q_u7(float v, float inv, int32_t zp) {
    int32_t q = (int32_t)nearbyintf(v * inv) + zp;
    if (q < 0) q = 0;
    if (q > 127) q = 127;
    return (uint8_t)q;
}

// x[n] contiguo -> xq[n] contiguo
EscalaAct cuantizar_u7(const float* x, int64_t n, uint8_t* xq) {
    float mn = 0.f, mx = 0.f;
    for (int64_t i = 0; i < n; ++i) {
        float v = x[i];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
    }
    EscalaAct e = rango_u7(mn, mx);
    if (e.zp == 0 && e.escala == 1.0f && mn == 0.f && mx == 0.f) {
        std::memset(xq, 0, (size_t)n);
        return e;
    }
    float inv = 1.0f / e.escala;
    for (int64_t i = 0; i < n; ++i) xq[i] = q_u7(x[i], inv, e.zp);
    return e;
}

// x[C,T] (disposicion de torch para conv) -> xq[T,C]: cuantiza y transpone en
// una pasada, porque la GEMM reduce sobre C y lo quiere contiguo.
EscalaAct cuantizar_u7_tr(const float* x, int64_t c, int64_t t, uint8_t* xq) {
    float mn = 0.f, mx = 0.f;
    for (int64_t i = 0; i < c * t; ++i) {
        float v = x[i];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
    }
    EscalaAct e = rango_u7(mn, mx);
    float inv = 1.0f / e.escala;
    for (int64_t ci = 0; ci < c; ++ci)
        for (int64_t ti = 0; ti < t; ++ti)
            xq[ti * c + ci] = q_u7(x[ci * t + ti], inv, e.zp);
    return e;
}

// -------------------------------------------------------------------- gemm

int32_t hsuma_epi32(__m256i v) {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v),
                              _mm256_extracti128_si256(v, 1));
    s = _mm_hadd_epi32(s, s);
    s = _mm_hadd_epi32(s, s);
    return _mm_cvtsi128_si32(s);
}

// Producto escalar de UNA fila de pesos contra TL filas de activaciones.
// El motivo de la plantilla: cada chunk de 32 bytes de peso se carga UNA vez
// y se usa contra las TL filas -- los pesos son el recurso caro (se leen de
// RAM), las activaciones caben en cache. TL fijo en compilacion para que los
// TL acumuladores vivan en registros (TL<=8: 8 acumuladores + peso + temporal
// caben en los 16 ymm).
template <int TL>
void gemm_lote(const uint8_t* xq, int64_t paso_x, const int8_t* wf,
               int64_t k, int32_t* sal) {
    __m256i acc[TL];
    for (int i = 0; i < TL; ++i) acc[i] = _mm256_setzero_si256();
    const __m256i unos = _mm256_set1_epi16(1);
    for (int64_t kk = 0; kk < k; kk += 32) {
        const __m256i wv =
            _mm256_loadu_si256((const __m256i*)(wf + kk));
        for (int i = 0; i < TL; ++i) {
            const __m256i xv = _mm256_loadu_si256(
                (const __m256i*)(xq + (int64_t)i * paso_x + kk));
            // xv sin signo (u7), wv con signo: el orden de maddubs importa.
            acc[i] = _mm256_add_epi32(
                acc[i], _mm256_madd_epi16(_mm256_maddubs_epi16(xv, wv), unos));
        }
    }
    for (int i = 0; i < TL; ++i) sal[i] = hsuma_epi32(acc[i]);
}

// y[T,N] = desescalar(xq[T,K] . wq[N,K]^T) (+ sesgo). Paralelo sobre N: cada
// hilo recorre filas de peso disjuntas, una sola lectura de pesos por llamada
// sea cual sea T.
void gemm_din(const uint8_t* xq, EscalaAct ea, const int8_t* w,
              const float* escalas, const int32_t* sumas_w,
              const float* sesgo, float* y, int64_t t, int64_t k, int64_t n) {
#pragma omp parallel for schedule(static)
    for (int64_t nn = 0; nn < n; ++nn) {
        const int8_t* wf = w + nn * k;
        const float f = ea.escala * escalas[nn];
        // y = f*acc - f*zp*sum(w) + b, con el termino constante fuera del
        // bucle de t. zp*sum(w) cabe de sobra: 127 * 8192*127 ~ 1,3e8.
        const float cte =
            (sesgo ? sesgo[nn] : 0.0f) - f * (float)(ea.zp * (int64_t)sumas_w[nn]);
        int32_t acc[8];
        int64_t tt = 0;
        for (; t - tt >= 8; tt += 8) {
            gemm_lote<8>(xq + tt * k, k, wf, k, acc);
            for (int i = 0; i < 8; ++i) y[(tt + i) * n + nn] = f * (float)acc[i] + cte;
        }
        const int64_t resto = t - tt;
        switch (resto) {
            case 7: gemm_lote<7>(xq + tt * k, k, wf, k, acc); break;
            case 6: gemm_lote<6>(xq + tt * k, k, wf, k, acc); break;
            case 5: gemm_lote<5>(xq + tt * k, k, wf, k, acc); break;
            case 4: gemm_lote<4>(xq + tt * k, k, wf, k, acc); break;
            case 3: gemm_lote<3>(xq + tt * k, k, wf, k, acc); break;
            case 2: gemm_lote<2>(xq + tt * k, k, wf, k, acc); break;
            case 1: gemm_lote<1>(xq + tt * k, k, wf, k, acc); break;
            default: break;
        }
        for (int64_t i = 0; i < resto; ++i) y[(tt + i) * n + nn] = f * (float)acc[i] + cte;
    }
}

}  // namespace

// ----------------------------------------------------------------- C ABI

extern "C" {

// Para que el envoltorio Python detecte un .so desfasado antes de usarlo.
int vv_version(void) { return 1; }

// y[T,N] = x[T,K] . W^T + sesgo
//   w:       int8 [N,K] fila a fila (la misma disposicion que nn.Linear)
//   escalas: fp32 [N] (escala simetrica por fila)
//   sumas_w: i32 [N] (suma de wq por fila, precalculada al empaquetar)
//   sesgo:   fp32 [N] o NULL
// Devuelve 0, -1 (formas invalidas: K no multiplo de 32) o -2 (sin memoria).
int vv_linear_din(const float* x, const int8_t* w, const float* escalas,
                  const int32_t* sumas_w, const float* sesgo, float* y,
                  int64_t t, int64_t k, int64_t n) {
    if (t <= 0 || n <= 0 || k <= 0 || (k & 31)) return -1;
    uint8_t* xq = (uint8_t*)std::malloc((size_t)(t * k));
    if (!xq) return -2;
    EscalaAct ea = cuantizar_u7(x, t * k, xq);
    gemm_din(xq, ea, w, escalas, sumas_w, sesgo, y, t, k, n);
    std::free(xq);
    return 0;
}

// Salida COMPLETA de conv_transpose1d(x[Cin,T], W, sesgo, stride=s), es decir
// y[Cout, (T-1)*s+k]: la misma que devuelve F.conv_transpose1d, para que el
// recorte causal del modulo que envuelve opere identico.
//
// COMPLETA A PROPOSITO, Y CUESTA: en streaming el que llama se queda solo con
// las ultimas T_nuevas*s posiciones, y a esas solo contribuyen las ultimas
// ceil(k/s) entradas. En la subida 2048->1024 (T=16 con contexto, k=16, s=8)
// eso son 2 de 16: se calcula 8x de mas. Se acepta porque asi la paridad
// contra F.conv_transpose1d es comprobable elemento a elemento, y porque el
// techo de la mejora es corto: los pesos de esa capa son ~2,6 ms de lectura
// a 13 GB/s frente a los 4,7 ms que tarda hoy. Sobra aritmetica, no bytes.
// Si algun dia se recorta, el envoltorio tendra que pasar cuantas posiciones
// finales necesita y la paridad pasara a comprobarse solo sobre ellas.
//
// Se calcula como GEMM (reduccion sobre Cin) mas dispersion col2im:
//   w:       int8 [Cout*k, Cin]: el W[Cin,Cout,k] de torch permutado a
//            (Cout,k,Cin) y aplanado, cada columna de salida contigua.
//   escalas: fp32 [Cout*k] (la escala por canal Cout, repetida k veces)
//   sumas_w: i32 [Cout*k]
//   sesgo:   fp32 [Cout] o NULL
int vv_convtr_din(const float* x, const int8_t* w, const float* escalas,
                  const int32_t* sumas_w, const float* sesgo, float* y,
                  int64_t t, int64_t cin, int64_t cout, int64_t k, int64_t s) {
    if (t <= 0 || cout <= 0 || k <= 0 || s <= 0) return -1;
    if (cin <= 0 || (cin & 31)) return -1;
    const int64_t largo = (t - 1) * s + k;
    const int64_t nc = cout * k;
    uint8_t* xq = (uint8_t*)std::malloc((size_t)(t * cin));
    float* g = (float*)std::malloc((size_t)(t * nc) * sizeof(float));
    if (!xq || !g) {
        std::free(xq);
        std::free(g);
        return -2;
    }
    EscalaAct ea = cuantizar_u7_tr(x, cin, t, xq);
    gemm_din(xq, ea, w, escalas, sumas_w, nullptr, g, t, cin, nc);
    // Dispersion: y[co, ti*s + j] += g[ti, co*k + j]. Paralela sobre co: cada
    // hilo es dueno de una fila entera de y, no hay carreras aunque las
    // ventanas de ti consecutivos se solapen (k > s).
#pragma omp parallel for schedule(static)
    for (int64_t co = 0; co < cout; ++co) {
        float* yc = y + co * largo;
        const float b = sesgo ? sesgo[co] : 0.0f;
        for (int64_t p = 0; p < largo; ++p) yc[p] = b;
        for (int64_t ti = 0; ti < t; ++ti) {
            const float* gf = g + ti * nc + co * k;
            float* yv = yc + ti * s;
            for (int64_t j = 0; j < k; ++j) yv[j] += gf[j];
        }
    }
    std::free(xq);
    std::free(g);
    return 0;
}

}  // extern "C"
