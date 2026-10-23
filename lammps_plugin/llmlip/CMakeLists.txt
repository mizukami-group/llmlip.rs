cmake_minimum_required(VERSION 3.20)
project(lammps_llmlip_plugin LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(LAMMPS_SOURCE_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../../third_party/lammps"
    CACHE PATH "Path to pinned LAMMPS checkout")
set(LLMLIP_CAPI_INCLUDE_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../../crates/llmlip-capi/include"
    CACHE PATH "Path to llmlip C ABI headers")
set(LLMLIP_CAPI_LIBRARY
    ""
    CACHE FILEPATH "Path to libllmlip_capi static/shared library")
option(LLMLIP_LAMMPS_USE_STUBS "Use LAMMPS STUBS mpi.h for syntax-only/non-MPI builds" ON)

add_library(lammps_llmlipplugin SHARED
  pair_llmlip.cpp
  lammpsplugin_registration.cpp
)

target_include_directories(lammps_llmlipplugin PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}"
  "${LAMMPS_SOURCE_DIR}/src"
  "${LLMLIP_CAPI_INCLUDE_DIR}"
)

if(LLMLIP_LAMMPS_USE_STUBS)
  target_include_directories(lammps_llmlipplugin PRIVATE "${LAMMPS_SOURCE_DIR}/src/STUBS")
endif()

if(LLMLIP_CAPI_LIBRARY)
  target_link_libraries(lammps_llmlipplugin PRIVATE "${LLMLIP_CAPI_LIBRARY}")
endif()

if(APPLE)
  target_link_options(lammps_llmlipplugin PRIVATE "-Wl,-undefined,dynamic_lookup")
endif()
