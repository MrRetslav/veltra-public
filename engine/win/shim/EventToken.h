// WebView2.h includes "EventToken.h"; MinGW-w64 ships it as eventtoken.h, which a case-sensitive cross-compile host
// (Linux) does not find. This shim forwards to the real header (include_next: no self-recursion on Windows file systems).
#pragma once
#include_next <eventtoken.h>
