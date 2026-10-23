cmake_minimum_required(VERSION 3.20)

project(llmlip_openmm_capi_bridge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest)

set(LLMLIP_CAPI_INCLUDE_DIR "" CACHE PATH "Directory containing llmlip_capi.h")
set(LLMLIP_CAPI_LIBRARY "" CACHE FILEPATH "Path to libllmlip_capi")
set(LLMLIP_CAPI_EXTRA_LINK_LIBS "" CACHE STRING "Additional native libraries required by the Rust llmlip-capi staticlib")
set(LLMLIP_CAPI_EXTRA_LINK_OPTIONS "" CACHE STRING "Additional native linker options required by the Rust llmlip-capi staticlib")
option(LLMLIP_OPENMM_BUILD_TESTS "Build llmlip OpenMM C API bridge tests" ${BUILD_TESTING})

find_package(Threads REQUIRED)

if(NOT LLMLIP_CAPI_INCLUDE_DIR)
  message(FATAL_ERROR "Set LLMLIP_CAPI_INCLUDE_DIR to crates/llmlip-capi/include")
endif()
if(NOT LLMLIP_CAPI_LIBRARY)
  message(FATAL_ERROR "Set LLMLIP_CAPI_LIBRARY to target/release/libllmlip_capi.a or .dylib/.so")
endif()
if(NOT EXISTS "${LLMLIP_CAPI_LIBRARY}")
  message(FATAL_ERROR "LLMLIP_CAPI_LIBRARY does not exist: ${LLMLIP_CAPI_LIBRARY}")
endif()

add_library(llmlip_openmm_capi_bridge STATIC
  src/llmlip_openmm_capi.cpp
)

target_include_directories(llmlip_openmm_capi_bridge
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${LLMLIP_CAPI_INCLUDE_DIR}"
)

target_link_libraries(llmlip_openmm_capi_bridge
  PUBLIC
    "${LLMLIP_CAPI_LIBRARY}"
    Threads::Threads
    ${CMAKE_DL_LIBS}
    ${LLMLIP_CAPI_EXTRA_LINK_LIBS}
)

if(UNIX AND NOT APPLE)
  target_link_libraries(llmlip_openmm_capi_bridge PUBLIC m)
endif()

target_link_options(llmlip_openmm_capi_bridge
  INTERFACE
    ${LLMLIP_CAPI_EXTRA_LINK_OPTIONS}
)

if(LLMLIP_OPENMM_BUILD_TESTS)
  add_executable(llmlip_openmm_capi_bridge_tests
    tests/llmlip_openmm_capi_tests.cpp
  )
  target_include_directories(llmlip_openmm_capi_bridge_tests
    PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/include"
      "${LLMLIP_CAPI_INCLUDE_DIR}"
  )
  add_test(
    NAME llmlip_openmm_capi_bridge_tests
    COMMAND llmlip_openmm_capi_bridge_tests
  )

  add_executable(llmlip_openmm_capi_link_smoke
    tests/llmlip_openmm_capi_link_smoke.cpp
  )
  target_link_libraries(llmlip_openmm_capi_link_smoke
    PRIVATE
      llmlip_openmm_capi_bridge
  )
  add_test(
    NAME llmlip_openmm_capi_link_smoke
    COMMAND llmlip_openmm_capi_link_smoke
  )
endif()
