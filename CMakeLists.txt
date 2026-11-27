cmake_minimum_required(VERSION 3.20)
# Veltra (formerly LittleBit-Turbo): bare-metal MoE inference engine.
project(Veltra LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(LB_BUILD_SERVER "Build the OpenAI-compatible HTTP server" ON)
option(LB_BUILD_TESTS  "Build self-test executables" ON)
option(VELTRA_EMBED_UI "Compile the web UI (chat/*.html) into the server binaries" ON)
option(VELTRA_CUDA     "Build the CUDA backend (engine/cuda, sm_86 / RTX 3060 by default)" OFF)

if(MSVC)
  # static CRT (/MT) for every target, so veltra.exe needs no VC++ redistributable
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

# x86-64 kernels use per-function target attributes + runtime CPUID dispatch, so no global
# -mavx2/-mavx512 flag is needed (one binary runs on any x86-64 and uses the best path).
set(LB_WARN -Wall -Wextra -Wno-unused-parameter)
set(LB_OPT  -O3 -fno-math-errno)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
  # dotprod + fp16 are guaranteed on every Snapdragon 8-series / modern Cortex-A7x/X cores
  list(APPEND LB_OPT -march=armv8.2-a+dotprod+fp16)
endif()

set(LB_CORE_SRC
  engine/binary_gemv.cpp
  engine/nibble_gemv.cpp
  engine/i3_gemv.cpp
  engine/q2k_gemv.cpp
  engine/q2k_selftest.cpp
  engine/kernels_x86.cpp
  engine/kernels_neon.cpp
  engine/selftest.cpp
  engine/lbt_model.cpp
  engine/ops.cpp
  engine/transformer.cpp
  engine/batch_forward.cpp
  engine/batch_gen.cpp
  engine/aux_dump.cpp
  engine/deltanet.cpp
  engine/iq_gemv.cpp
  engine/hybrid_forward.cpp
  engine/exit_probe.cpp
  engine/speculative.cpp
  engine/rest_index.cpp
  engine/model_config.cpp
  engine/engine.cpp
  engine/tokenizer.cpp
  engine/chat_template.cpp
  engine/platform.cpp
  engine/backend.cpp
  engine/search.cpp
  engine/search_verifier.cpp
  engine/q4k_quant.cpp
  engine/budget_probe.cpp
)
add_library(littlebit_core STATIC ${LB_CORE_SRC})
target_include_directories(littlebit_core PUBLIC engine)
find_package(Threads REQUIRED)
target_link_libraries(littlebit_core PUBLIC Threads::Threads)
target_compile_options(littlebit_core PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${LB_WARN};${LB_OPT}>")

# Binaries were renamed lb_selftest / lb_gen / lb_server -> veltra-selftest / veltra-gen / veltra-server (25 Nov 2026).
# On non-Windows hosts a post-build step also creates the old names as symlinks next to the new binaries.
function(veltra_compat_link target old_name)
  if(NOT WIN32)
    add_custom_command(TARGET ${target} POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE_NAME:${target}> $<TARGET_FILE_DIR:${target}>/${old_name}
      VERBATIM)
  endif()
endfunction()

if(LB_BUILD_TESTS)
  add_executable(veltra-selftest engine/selftest_main.cpp)
  target_link_libraries(veltra-selftest PRIVATE littlebit_core)
  target_compile_options(veltra-selftest PRIVATE ${LB_WARN} ${LB_OPT})
  veltra_compat_link(veltra-selftest lb_selftest)
  add_executable(veltra-gen engine/lb_gen.cpp)
  target_link_libraries(veltra-gen PRIVATE littlebit_core)
  target_compile_options(veltra-gen PRIVATE ${LB_WARN} ${LB_OPT})
  veltra_compat_link(veltra-gen lb_gen)
endif()

# ---- web UI -> generated/embedded_ui.h (served by veltra-server / veltra.exe when no --web-dir is given)
# "path@name" embeds path under another name: the desktop app UI is served as desktop.html (GET /desktop, GET / with --ui desktop)
set(VELTRA_UI_FILES chat/index.html chat/benchmarks.html chat/arena.html chat/chat.html chat/benchmarks_live.json
    desktop_ui/index.html@desktop.html desktop_ui/overlay.html@overlay.html desktop_ui/pro.js desktop_ui/pro.css desktop_ui/market.js)
set(VELTRA_GEN_DIR ${CMAKE_BINARY_DIR}/generated)
if(VELTRA_EMBED_UI)
  set(_ui_deps "")
  foreach(f IN LISTS VELTRA_UI_FILES)
    string(REGEX REPLACE "@.*$" "" f "${f}")
    list(APPEND _ui_deps ${CMAKE_SOURCE_DIR}/${f})
  endforeach()
  string(REPLACE ";" "," _ui_arg "${VELTRA_UI_FILES}")
  add_custom_command(
    OUTPUT ${VELTRA_GEN_DIR}/embedded_ui.h
    COMMAND ${CMAKE_COMMAND} -E make_directory ${VELTRA_GEN_DIR}
    COMMAND ${CMAKE_COMMAND} -DOUT=${VELTRA_GEN_DIR}/embedded_ui.h -DROOT=${CMAKE_SOURCE_DIR} -DFILES=${_ui_arg}
            -P ${CMAKE_SOURCE_DIR}/cmake/embed_ui.cmake
    DEPENDS ${_ui_deps} ${CMAKE_SOURCE_DIR}/cmake/embed_ui.cmake
    COMMENT "Embedding web UI into generated/embedded_ui.h"
    VERBATIM)
  add_custom_target(veltra_ui DEPENDS ${VELTRA_GEN_DIR}/embedded_ui.h)
endif()

# ---- optional CUDA backend (engine/cuda): -DVELTRA_CUDA=ON [-DCMAKE_CUDA_ARCHITECTURES=86]
if(VELTRA_CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)   # Ampere GA106: RTX 3060
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  # compiled into littlebit_core itself (the CUDA backend calls back into the CPU kernels for its fallback layers)
  target_sources(littlebit_core PRIVATE engine/cuda/cuda_backend.cu engine/cuda/veltra_moe_grouped_gemv.cu)
  target_compile_definitions(littlebit_core PUBLIC VELTRA_CUDA=1)
  target_compile_options(littlebit_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -lineinfo>)
  target_link_libraries(littlebit_core PUBLIC CUDA::cudart_static CUDA::cublas)
endif()

set(VELTRA_SERVER_SRC engine/server.cpp engine/hw_detect.cpp engine/model_hub.cpp engine/catalog.cpp engine/hf_convert.cpp engine/cluster.cpp
    engine/relay.cpp engine/remote_client.cpp engine/test_runner.cpp)
function(veltra_server_target name)
  add_executable(${name} ${VELTRA_SERVER_SRC})
  target_link_libraries(${name} PRIVATE littlebit_core ${CMAKE_DL_LIBS})
  if(WIN32)
    target_link_libraries(${name} PRIVATE ws2_32 wininet shell32 psapi bcrypt)   # dxgi.dll / nvml.dll are loaded at run time (LoadLibraryA)
  endif()
  if(VELTRA_EMBED_UI)
    add_dependencies(${name} veltra_ui)
    target_include_directories(${name} PRIVATE ${VELTRA_GEN_DIR})
    target_compile_definitions(${name} PRIVATE VELTRA_EMBED_UI=1)
  endif()
  target_compile_options(${name} PRIVATE ${LB_WARN} ${LB_OPT})
endfunction()

if(LB_BUILD_SERVER)
  veltra_server_target(veltra-server)
  veltra_compat_link(veltra-server lb_server)
  if(WIN32)
    # veltra.exe: native desktop app (engine/win_app.cpp: Win32 window + WebView2, server in-process on an ephemeral
    # loopback port); `veltra.exe --headless` = veltra-server. Single file: UI embedded, static runtime, no WebView2Loader.dll.
    veltra_server_target(veltra)
    target_sources(veltra PRIVATE engine/win_app.cpp engine/win/veltra.rc)
    set_target_properties(veltra PROPERTIES WIN32_EXECUTABLE ON)
    target_compile_definitions(veltra PRIVATE VELTRA_STANDALONE=1 VELTRA_DESKTOP=1 _WIN32_WINNT=0x0A00)
    target_include_directories(veltra PRIVATE third_party/webview2 engine/win)   # engine/win: veltra.rc finds its .ico/.manifest
    if(NOT MSVC)
      target_include_directories(veltra PRIVATE engine/win/shim)   # "EventToken.h" -> MinGW's eventtoken.h
    endif()
    target_link_libraries(veltra PRIVATE ole32 oleaut32 uuid dwmapi user32 gdi32 shell32 advapi32 bcrypt)
    if(MINGW)
      target_link_options(veltra PRIVATE -static -static-libgcc -static-libstdc++)
    endif()
  endif()
endif()

# Shared library with the plain C API (python ctypes tests, Android JNI links the same sources).
add_library(littlebit_capi SHARED engine/capi.cpp)
target_link_libraries(littlebit_capi PRIVATE littlebit_core)
target_compile_options(littlebit_capi PRIVATE ${LB_WARN} ${LB_OPT})
set_target_properties(littlebit_capi PROPERTIES PREFIX "")
