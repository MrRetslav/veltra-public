// Plain C API over lb::Engine. Used by the Android JNI bridge and python ctypes tests.
// All functions are thread-compatible per handle (one generation at a time per handle).
#pragma once
#include <stdint.h>

#ifdef _WIN32
#define LB_API __declspec(dllexport)
#else
#define LB_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef void* lb_handle;

// Called per generated token. Return 0 to continue, non-zero to cancel.
typedef int (*lb_token_cb)(int token_id, const char* utf8_piece, void* user);

// --- engine -------------------------------------------------------------------------------
LB_API lb_handle lb_open(const char* lbt_path, char* err, int err_len);   // NULL on failure
LB_API void lb_close(lb_handle h);
LB_API const char* lb_isa_name(lb_handle h);
LB_API const char* lb_config_json(lb_handle h);                           // valid until lb_close

// messages_json: [{"role":"user","content":"..."}, ...]. Returns number of tokens written
// (or required size if out is NULL / too small -> negative of required count).
LB_API int lb_chat_tokens(lb_handle h, const char* messages_json, int32_t* out, int cap);
LB_API int lb_encode(lb_handle h, const char* text, int add_special, int32_t* out, int cap);
LB_API int lb_decode(lb_handle h, const int32_t* ids, int n, char* out, int cap);

// opts_json (all optional): {"max_new_tokens":256,"temperature":0,"top_p":1,"top_k":0,"seed":0,
//   "speculative":true,"ngram":true,"layer_skip":true,"k_min":2,"k_max":8,"n_threads":0}
// Returns generated token count (>=0) or negative on error. stats_json (optional, may be NULL) receives
// {"prompt_tokens","generated_tokens","prefill_s","decode_s","tokens_per_s","drafted","accepted",...}.
LB_API int lb_generate(lb_handle h, const int32_t* prompt, int n_prompt, const char* opts_json,
                       lb_token_cb cb, void* user, char* stats_json, int stats_cap);
LB_API void lb_reset(lb_handle h);

// --- verification helpers (used by tests/verify_precision.py) ---------------------------------
// Runs one LittleBitLinear layer `name` ("blk.0.q" or any prefix with <name>.prim.* tensors) on x[T][n]
// producing y[T][m]. use_res=0 -> primary path only. isa: -1 auto, 0 scalar, 1 avx2, 2 avx512, 3 neon.
LB_API int lb_linear_forward(lb_handle h, const char* name, const float* x, int T, float* y, int use_res, int isa);
// Same with an engine thread pool of `threads` workers (<= 1: single thread). Grid linears: lb_grid_info -> planes P, group G (128 or 0).
LB_API int lb_linear_forward_mt(lb_handle h, const char* name, const float* x, int T, float* y, int use_res, int isa, int threads);
LB_API int lb_grid_info(lb_handle h, const char* name, int* planes, int* group);
LB_API int lb_linear_dims(lb_handle h, const char* name, int* m, int* n, int* r_prim, int* r_res);
// Bit-exact kernel self-test; returns number of failures, report written to out.
LB_API int lb_kernel_selftest(char* out, int cap);
LB_API int lb_isa_available(int isa);   // 1 if the kernel for this ISA id can run on this CPU

#ifdef __cplusplus
}
#endif
