// engine/cluster.h - compute selection (GPU / CPU / cluster) and a tiny blocking HTTP/1.1 client for cluster nodes.
// Used by veltra-server for GET /api/compute, POST /api/compute/select and request-level chat forwarding.
//
// Honest scope: "cluster" means request-level routing (a whole chat request goes to one online node, round-robin).
// Splitting one model's layers over machines is NOT implemented; layer_plan() only proposes a split.
// No OS header leaks from here: sockets stay inside engine/cluster.cpp.
#pragma once
#include <cstddef>
#include <string>
#include <vector>

namespace veltra::cluster {

constexpr int kMaxNodes = 8;
constexpr int kPingTimeoutMs = 1500;

struct Selection {
    std::string mode = "cpu";          // "gpu" | "cpu" | "cluster"
    std::vector<int> gpu_indices;      // indices into detect_hardware().gpus
    std::vector<std::string> nodes;    // "host:port", host = hostname / IPv4 / [IPv6]
};

struct NodeStatus {
    std::string node;
    bool online = false;
    double latency_ms = 0;
    std::string model;   // "model" of the node's /health JSON ("" when absent)
    std::string error;   // why it is offline
};

struct HttpResult {
    int status = 0;            // 0 = transport failure (see err)
    std::string content_type;
    std::string body;          // de-chunked
    std::string err;
    double ms = 0;             // wall time of the whole exchange
};

// "host:port" -> host (brackets removed for IPv6) + port 1..65535. False + err when invalid.
bool parse_node(const std::string& s, std::string* host, int* port, std::string* err);

// Validates a POST /api/compute/select body against `n_gpus` detected GPUs. Duplicate nodes are dropped.
bool parse_selection(const std::string& body, int n_gpus, Selection& out, std::string& err);

// Current selection (loaded once from <app data>/compute.json; default {"mode":"cpu"}).
Selection current();
// Replaces the selection in memory and writes compute.json. Returns "" on success, else a warning
// (the in-memory selection is still updated when only persisting failed).
std::string set_current(const Selection& s);
std::string selection_json(const Selection& s);   // {"mode":..,"gpu_indices":[..],"nodes":[..]}

// Blocking HTTP/1.1 request with an overall deadline (connect + send + receive). `extra_headers` lines end in \r\n.
HttpResult http_request(const std::string& host, int port, const std::string& method, const std::string& path,
                        const std::string& body, const std::string& extra_headers, int timeout_ms, size_t max_bytes);

// GET http://node/health with kPingTimeoutMs; all nodes in parallel. Results also refresh the routing cache.
std::vector<NodeStatus> ping_all(const std::vector<std::string>& nodes);
std::string node_status_json(const NodeStatus& s);

// Next online node of `nodes`, round-robin; uses ping results younger than a few seconds, pings otherwise.
// "" when none is online.
std::string pick_online_node(const std::vector<std::string>& nodes);
// Marks `node` offline in the routing cache after a transport failure (until the next ping).
void mark_offline(const std::string& node);

}  // namespace veltra::cluster
