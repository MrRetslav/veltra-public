# Windows dev toolchain: zig-as-clang (pip install ziglang cmake ninja). Use:
#   cmake -S . -B build -G Ninja -DCMAKE_TOOLCHAIN_FILE=cmake/zig-windows.cmake
set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_C_COMPILER   ${CMAKE_CURRENT_LIST_DIR}/../scripts/zigcc.cmd)
set(CMAKE_CXX_COMPILER ${CMAKE_CURRENT_LIST_DIR}/../scripts/zigcxx.cmd)
set(CMAKE_AR     ${CMAKE_CURRENT_LIST_DIR}/../scripts/zigar.cmd CACHE FILEPATH "")
set(CMAKE_RANLIB ${CMAKE_CURRENT_LIST_DIR}/../scripts/zigranlib.cmd CACHE FILEPATH "")
