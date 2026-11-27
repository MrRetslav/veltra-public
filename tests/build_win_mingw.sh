#!/bin/bash
# Cross-compile the Windows desktop app veltra.exe on Linux with MinGW-w64 (posix thread model, static runtime).
#   apt-get install g++-mingw-w64-x86-64-posix   (Debian/Ubuntu; provides x86_64-w64-mingw32-g++-posix + windres)
#   bash tests/build_win_mingw.sh <repo dir> [out dir]
# Mirrors CMake's `veltra` target (VELTRA_STANDALONE + VELTRA_DESKTOP + VELTRA_EMBED_UI, win_app.cpp + veltra.rc,
# GUI subsystem, -static). On Windows itself use CMake + MinGW (docs/WINDOWS_BUILD.md):
#   cmake -B build -G "MinGW Makefiles" && cmake --build build --target veltra
set -e
D=${1:-.}
cd "$D"
OUT=${2:-build-win}
OBJ=$OUT/obj
mkdir -p $OBJ $OUT/generated
CXX=${CXX:-x86_64-w64-mingw32-g++-posix}
command -v $CXX >/dev/null || CXX=x86_64-w64-mingw32-g++
FLAGS="-std=c++20 -O3 -fno-math-errno -Iengine -I$OUT/generated -Ithird_party/webview2 -Iengine/win/shim -Wno-unused-function -D_WIN32_WINNT=0x0A00"
cmake -DOUT=$OUT/generated/embedded_ui.h -DROOT=$PWD \
  -DFILES=chat/index.html,chat/benchmarks.html,chat/arena.html,chat/chat.html,chat/benchmarks_live.json,desktop_ui/index.html@desktop.html,desktop_ui/overlay.html@overlay.html,desktop_ui/pro.js,desktop_ui/pro.css,desktop_ui/market.js -P cmake/embed_ui.cmake
SRC="binary_gemv nibble_gemv i3_gemv q2k_gemv q2k_selftest kernels_x86 kernels_neon selftest lbt_model ops transformer batch_forward batch_gen aux_dump exit_probe deltanet iq_gemv hybrid_forward speculative rest_index model_config engine tokenizer chat_template platform backend search search_verifier q4k_quant budget_probe hw_detect model_hub catalog hf_convert cluster relay remote_client test_runner server win_app"
pids=""
for f in $SRC; do
  if [ ! -f $OBJ/$f.o ] || [ engine/$f.cpp -nt $OBJ/$f.o ] || [ -n "$(find engine $OUT/generated -name '*.h' -newer $OBJ/$f.o | head -1)" ]; then
    echo "compile $f"
    nice -n 10 $CXX $FLAGS -DVELTRA_STANDALONE=1 -DVELTRA_DESKTOP=1 -DVELTRA_EMBED_UI=1 -c engine/$f.cpp -o $OBJ/$f.o &
    pids="$pids $!"
    while [ "$(jobs -r | wc -l)" -ge 4 ]; do sleep 0.2; done
  fi
done
for p in $pids; do wait $p; done   # fails the script on the first failed compile
OBJS=""; for f in $SRC; do OBJS="$OBJS $OBJ/$f.o"; done
# icon + application manifest (per-monitor-v2 DPI, Common Controls v6, UTF-8 code page) + version info
RC=${WINDRES:-x86_64-w64-mingw32-windres}
$RC -I engine/win engine/win/veltra.rc -O coff -o $OBJ/veltra_res.o
nice -n 10 $CXX -static -static-libgcc -static-libstdc++ -mwindows -o $OUT/veltra.exe $OBJS $OBJ/veltra_res.o   -lws2_32 -lwininet -lshell32 -lpsapi -lole32 -loleaut32 -luuid -ldwmapi -luser32 -lgdi32 -ladvapi32 -lbcrypt
ls -la $OUT/veltra.exe
# imported DLLs: system DLLs only (no libstdc++/libgcc/libwinpthread)
x86_64-w64-mingw32-objdump -p $OUT/veltra.exe | grep "DLL Name" | sort -u
