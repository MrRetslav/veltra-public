// Textually included three times from nibble_gemv.cpp with NB_FN / NB_ATTR / NB_MODE defined (0: AVX2 maddubs, 1: ymm vpdpbusd, 2: zmm vpdpbusd).
// (g++ cannot inline always_inline intrinsics into a body whose target attribute differs, so each ISA gets its own copy.)
template <int TG> NB_ATTR static void NB_FN(const NibView& v, int rb, const int8_t* q, size_t qs, const int32_t* A, const float* ss, int e0,
                                      float* yE, size_t yst) {
    const int nt = v.ntile;
    alignas(32) float acc[TG][8][8];
    std::memset(acc, 0, sizeof acc);
    const __m256i m0f = _mm256_set1_epi8(15), ones = _mm256_set1_epi16(1);
    const uint8_t* wb = v.w + (size_t)rb * nt * kNibTileBytes;
    const float* gb = v.gs + (size_t)rb * nt * 64;
    for (int t = 0; t < nt; t++) {
        const uint8_t* wt = wb + (size_t)t * kNibTileBytes;
        const float* g = gb + (size_t)t * 64;
        const int8_t* ap[TG];
        int a15[TG];
        alignas(32) float gk[TG][64];
        const float* gp[TG];
        for (int k = 0; k < TG; k++) {
            ap[k] = q + (size_t)(e0 + k) * qs + (size_t)t * 128;
            a15[k] = 15 * A[(size_t)(e0 + k) * nt + t];
            if (ss) { const float sk = ss[(size_t)(e0 + k) * nt + t]; for (int r = 0; r < 64; r++) gk[k][r] = g[r] * sk; gp[k] = gk[k]; }
            else gp[k] = g;
        }
        for (int rg = 0; rg < 8; rg++) {
            __m256i s[TG][8];
            for (int j = 0; j < 8; j++) {
                const uint8_t* wr = wt + (rg * 8 + j) * 64;
#if NB_MODE == 2
                {
                    const __m512i w = _mm512_loadu_si512((const void*)wr);
                    const __m512i m512 = _mm512_set1_epi8(15);
                    const __m512i lo = _mm512_and_si512(w, m512), hi = _mm512_and_si512(_mm512_srli_epi16(w, 4), m512);
                    for (int k = 0; k < TG; k++) {
                        __m512i d = _mm512_dpbusd_epi32(_mm512_dpbusd_epi32(_mm512_setzero_si512(), lo, _mm512_loadu_si512((const void*)ap[k])), hi,
                                                        _mm512_loadu_si512((const void*)(ap[k] + 64)));
                        s[k][j] = _mm256_add_epi32(_mm512_castsi512_si256(d), _mm512_extracti64x4_epi64(d, 1));
                    }
                }
#else
                {
                    const __m256i wx = _mm256_loadu_si256((const __m256i*)wr), wy = _mm256_loadu_si256((const __m256i*)(wr + 32));
                    const __m256i lx = _mm256_and_si256(wx, m0f), ly = _mm256_and_si256(wy, m0f);
                    const __m256i hx = _mm256_and_si256(_mm256_srli_epi16(wx, 4), m0f), hy = _mm256_and_si256(_mm256_srli_epi16(wy, 4), m0f);
                    for (int k = 0; k < TG; k++) {
                        const __m256i a0 = _mm256_loadu_si256((const __m256i*)(ap[k])), a1 = _mm256_loadu_si256((const __m256i*)(ap[k] + 32));
                        const __m256i a2 = _mm256_loadu_si256((const __m256i*)(ap[k] + 64)), a3 = _mm256_loadu_si256((const __m256i*)(ap[k] + 96));
#if NB_MODE == 0
                        {
                            const __m256i p = _mm256_add_epi16(_mm256_add_epi16(_mm256_maddubs_epi16(lx, a0), _mm256_maddubs_epi16(ly, a1)),
                                                               _mm256_add_epi16(_mm256_maddubs_epi16(hx, a2), _mm256_maddubs_epi16(hy, a3)));
                            s[k][j] = _mm256_madd_epi16(p, ones);
                        }
#else
                        {
                            __m256i d = _mm256_dpbusd_epi32(_mm256_setzero_si256(), lx, a0);
                            d = _mm256_dpbusd_epi32(d, ly, a1);
                            d = _mm256_dpbusd_epi32(d, hx, a2);
                            s[k][j] = _mm256_dpbusd_epi32(d, hy, a3);
                        }
#endif
                    }
                }
#endif
            }
            for (int k = 0; k < TG; k++) {
                const __m256i z = _mm256_sub_epi32(_mm256_slli_epi32(nb_red8(s[k]), 1), _mm256_set1_epi32(a15[k]));
                _mm256_store_ps(acc[k][rg], _mm256_fmadd_ps(_mm256_cvtepi32_ps(z), _mm256_loadu_ps(gp[k] + rg * 8), _mm256_load_ps(acc[k][rg])));
            }
        }
    }
    for (int k = 0; k < TG; k++) std::memcpy(yE + (size_t)(e0 + k) * yst + (size_t)rb * 64, acc[k], 64 * sizeof(float));
}

