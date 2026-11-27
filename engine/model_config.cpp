#include "model_config.h"

#include "json.h"

namespace lb {

bool parse_model_config(const LbtFile& f, ModelConfig* c, std::string* err) {
    json::Value m;
    std::string e;
    if (!json::parse(f.meta_json(), &m, &e)) { if (err) *err = "bad meta json: " + e; return false; }
    auto geti = [&](const char* k, int d) { return (int)m[k].as_int(d); };
    c->arch = m["arch"].as_string("llama");
    c->model_name = m["model_name"].as_string("");
    c->hidden_size = geti("hidden_size", 0);
    c->num_layers = geti("num_layers", 0);
    c->num_heads = geti("num_heads", 0);
    c->num_kv_heads = geti("num_kv_heads", c->num_heads);
    c->head_dim = geti("head_dim", c->num_heads ? c->hidden_size / c->num_heads : 0);
    c->intermediate_size = geti("intermediate_size", 0);
    c->vocab_size = geti("vocab_size", 0);
    c->num_experts = geti("num_experts", 0);
    c->num_experts_per_tok = geti("num_experts_per_tok", 0);
    c->moe_intermediate_size = geti("moe_intermediate_size", 0);
    c->norm_topk_prob = m["norm_topk_prob"].as_bool(true);
    if (c->num_experts > 0 && (c->num_experts_per_tok <= 0 || c->num_experts_per_tok > c->num_experts || c->moe_intermediate_size <= 0)) {
        if (err) *err = "inconsistent MoE meta (num_experts_per_tok / moe_intermediate_size)";
        return false;
    }
    if (m.find("layer_types")) {   // hybrid (Gated DeltaNet + full attention) layout
        for (auto& v : m["layer_types"].as_array()) {
            const std::string t = v.as_string("");
            if (t != "full_attention" && t != "linear_attention") { if (err) *err = "unknown layer type '" + t + "'"; return false; }
            c->layer_full.push_back(t == "full_attention" ? 1 : 0);
        }
        if ((int)c->layer_full.size() != c->num_layers) { if (err) *err = "layer_types length != num_layers"; return false; }
    }
    c->rotary_dim = geti("rotary_dim", 0);
    c->attn_output_gate = m["attn_output_gate"].as_bool(false);
    c->lin_k_heads = geti("linear_num_key_heads", 0);
    c->lin_v_heads = geti("linear_num_value_heads", 0);
    c->lin_k_dim = geti("linear_key_head_dim", 0);
    c->lin_v_dim = geti("linear_value_head_dim", 0);
    c->lin_conv = geti("linear_conv_kernel_dim", 4);
    c->shared_expert_inter = geti("shared_expert_intermediate_size", 0);
    if (c->hybrid()) {
        bool any_dn = false;
        for (auto f : c->layer_full) any_dn = any_dn || !f;
        if (any_dn && (c->lin_k_heads <= 0 || c->lin_v_heads <= 0 || c->lin_v_heads % c->lin_k_heads != 0 || c->lin_k_dim <= 0 || c->lin_v_dim <= 0 || c->lin_conv < 2 || c->lin_conv > 8)) {
            if (err) *err = "inconsistent DeltaNet meta (linear_num_key_heads / linear_num_value_heads / linear_*_head_dim / linear_conv_kernel_dim)";
            return false;
        }
        if (c->rotary_dim > c->head_dim || c->rotary_dim % 2) { if (err) *err = "bad rotary_dim"; return false; }
    }
    c->max_position_embeddings = geti("max_position_embeddings", 4096);
    c->rope_theta = (float)m["rope_theta"].as_double(10000.0);
    c->rms_norm_eps = (float)m["rms_norm_eps"].as_double(1e-6);
    c->attn_scale = (float)m["attn_scale"].as_double(0.0);
    if (!(c->attn_scale >= 0.f && c->attn_scale <= 1.f)) { if (err) *err = "attn_scale out of range"; return false; }
    if (c->arch == "granite" && c->attn_scale <= 0.f) { if (err) *err = "granite model without attn_scale"; return false; }
    c->tie_word_embeddings = m["tie_word_embeddings"].as_bool(false);
    c->qk_norm = m["qk_norm"].as_bool(false);
    c->attn_bias = m["attn_bias"].as_bool(false);
    c->bos_token_id = geti("bos_token_id", -1);
    for (auto& v : m["eos_token_ids"].as_array()) c->eos_token_ids.push_back((int)v.as_int());
    c->chat_template = m["chat_template"].as_string("");
    if (m.find("draft_layer_skip")) {
        for (auto& v : m["draft_layer_skip"].as_array()) c->draft_layer_skip.push_back((int)v.as_int());
    } else {
        for (int i = 0; i < 64; i++) if ((f.header().draft_layer_mask >> i) & 1) c->draft_layer_skip.push_back(i);
    }
    if (c->hidden_size <= 0 || c->num_layers <= 0 || c->num_heads <= 0 || c->vocab_size <= 0 || (c->intermediate_size <= 0 && c->num_experts <= 0)) {
        if (err) *err = "meta is missing model dimensions (not a full model file?)";
        return false;
    }
    return true;
}

}  // namespace lb
