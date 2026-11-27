// veltra-selftest [--model x.vlt [--prompt-len 300] [--steps 12]]: kernel and verifier self-tests; with a model also the shared-prefix batch test.
#include <cstdio>
#include <cstdlib>
#include <string>

#include "deltanet.h"
#include "iq_gemv.h"
#include "q2k_selftest.h"
#include "search.h"
#include "selftest.h"

int main(int argc, char** argv) {
    std::string model, report;
    int prompt_len = 300, steps = 12;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--model") model = argv[i + 1];
        else if (a == "--prompt-len") prompt_len = std::atoi(argv[i + 1]);
        else if (a == "--steps") steps = std::atoi(argv[i + 1]);
    }
    int f = lb::run_kernel_selftest(report);
    f += lb::run_qk_selftest(report);
    f += lb::run_deltanet_selftest(report);
    f += lb::run_iq_selftest(report);
    f += lb::search_verifier_selftest(&report);
    if (!model.empty()) {
        try {
            f += lb::run_shared_prefix_selftest(model, prompt_len, steps, report);
        } catch (const std::exception& e) {
            report += std::string("shared-prefix selftest: ") + e.what() + "\n";
            f++;
        }
    }
    std::fputs(report.c_str(), stdout);
    return f == 0 ? 0 : 1;
}
