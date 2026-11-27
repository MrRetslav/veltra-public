// --teal-sparsity calibration: per-layer thresholds from a short built-in text (original filler written for this purpose: prose, Dutch, code, JSON,
// arithmetic). It contains no evaluation prompt and no evaluation answer.
#pragma once
#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include "tokenizer.h"
#include "transformer.h"

namespace lb {

inline const char* teal_calibration_text() {
    return "The harbour museum opens at nine in the morning. Visitors walk past old anchors, rope winches and a painted rowing boat "
           "that once carried mail between two islands. A guide explains how tide tables were copied by hand and why a small error "
           "could leave a cargo ship waiting for half a day. In the library upstairs, students compare maps from different centuries "
           "and notice that the coastline has slowly moved to the east.\n\n"
           "Vanmiddag regent het licht in de stad. Op de markt verkoopt een bakker warme broodjes, en iemand repareert een fiets naast "
           "de fontein. Morgen wordt het droger, zodat de buren eindelijk hun schutting kunnen verven.\n\n"
           "```python\ndef running_mean(values):\n    total = 0.0\n    for index, value in enumerate(values, start=1):\n"
           "        total += value\n        yield total / index\n```\n\n"
           "{\"library\":\"north\",\"shelves\":14,\"open\":true,\"visitors\":[12,31,8],\"note\":\"closed on public holidays\"}\n\n"
           "A train that covers 240 kilometres in three hours travels at 80 kilometres per hour. If the return trip takes four hours, "
           "the average speed over both journeys is the total distance divided by the total time.\n\n"
           "To prepare the soup, soften an onion in oil, add diced carrots and lentils, pour in the stock and let everything simmer "
           "until the lentils are tender. Season at the end, because the liquid reduces while it cooks.";
}

// Runs the calibration text through the model and switches TEAL on. Leaves the main KV cache empty.
inline void teal_calibrate(Transformer& model, const Tokenizer& tok, float P, int max_ctx) {
    const std::vector<int> ids = tok.encode(teal_calibration_text(), false);
    const int n = std::min((int)ids.size(), max_ctx - 1);
    model.teal_begin_calibration();
    for (int i = 0; i < n; i += 64) {
        const int c = std::min(64, n - i);
        ForwardOpts fo;
        fo.logits_from = c;
        model.forward(ids.data() + i, c, i, fo, nullptr);
    }
    model.teal_finish_calibration(P);
    model.kv().len = 0;
    std::fprintf(stderr, "[veltra] teal: thresholds calibrated on %d tokens (P=%.2f)\n", n, P);
}

}  // namespace lb
