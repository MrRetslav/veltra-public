// Veltra (formerly LittleBit-Turbo) common helpers. Zero dependencies (C++20 standard library only).
#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace lb {

inline float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t man = h & 0x3ff;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {  // subnormal -> normalise
            int e = -1;
            do { e++; man <<= 1; } while (!(man & 0x400));
            man &= 0x3ff;
            bits = sign | ((uint32_t)(127 - 15 - e) << 23) | (man << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (man << 13);
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

inline float bf16_to_f32(uint16_t h) {
    uint32_t bits = (uint32_t)h << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

// Round-to-nearest-even float -> IEEE half.
inline uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = (int32_t)((x >> 23) & 0xff) - 127 + 15;
    uint32_t man = x & 0x7fffffu;
    if (((x >> 23) & 0xff) == 0xff) return (uint16_t)(sign | 0x7c00u | (man ? 0x200u : 0));
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        man |= 0x800000u;
        int shift = 14 - exp;
        uint32_t half = man >> shift;
        uint32_t rem = man & ((1u << shift) - 1);
        uint32_t mid = 1u << (shift - 1);
        if (rem > mid || (rem == mid && (half & 1))) half++;
        return (uint16_t)(sign | half);
    }
    uint32_t half = sign | ((uint32_t)exp << 10) | (man >> 13);
    uint32_t rem = man & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (half & 1))) half++;
    return (uint16_t)half;
}

[[noreturn]] inline void fail(const std::string& msg) { throw std::runtime_error(msg); }

// Environment knob lookup: `name` is the suffix ("PIN", "PROFILE", ...). VELTRA_<name> wins; the historic
// LB_<name> is the fallback, so setups that only set LB_* behave exactly as before. No state, no allocation:
// safe inside static initialisers of any translation unit.
inline const char* env_get(const char* name) {
    char buf[128];
    const size_t n = std::strlen(name);
    if (n + 8 > sizeof buf) return nullptr;
    std::memcpy(buf, "VELTRA_", 7);
    std::memcpy(buf + 7, name, n + 1);
    if (const char* v = std::getenv(buf)) return v;
    std::memcpy(buf, "LB_", 3);
    std::memcpy(buf + 3, name, n + 1);
    return std::getenv(buf);
}

}  // namespace lb
