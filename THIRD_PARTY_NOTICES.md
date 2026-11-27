# Third-party notices

Veltra is licensed under the Apache License 2.0 (see `LICENSE`). It includes or derives from the following third-party material.

## llama.cpp / ggml (MIT)

`engine/iq_tables.h` contains lookup tables generated from llama.cpp's `ggml-common.h`; the reference dequantisers of the i-quants and the
Q2_K / Q3_K / Q4_K superblock layouts (transcoded from GGUF files) follow ggml's formats. The SIMD kernels are Veltra's own.

```
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Microsoft Edge WebView2 SDK header (BSD-style)

`third_party/webview2/WebView2.h` (Windows desktop app only). Copyright (C) Microsoft Corporation. The licence text is in
`third_party/webview2/LICENSE.txt`.

## Unicode Character Database

`engine/unicode_tables.h` is generated from the Unicode Character Database (`scripts/gen_unicode_tables.py`).
Copyright (c) 1991-2026 Unicode, Inc. Distributed under the Unicode License V3: https://www.unicode.org/license.txt

## Models

No model weights are part of this repository. Model files keep the licence of their authors. The Qwen3 and Qwen3.6 models used in the
benchmarks are published under Apache-2.0 by the Qwen team; check the model card of every file you download.
