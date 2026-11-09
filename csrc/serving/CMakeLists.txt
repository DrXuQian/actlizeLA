cmake_minimum_required(VERSION 3.19)
project(actlize_la_serving LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
get_filename_component(ALA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(ALA_TARGET "" CACHE STRING "Explicit serving target: ppu10, ppu15, cuda_sm120")
option(ALA_PPU10_PARALLEL_WU_EXPERIMENT "Build an opt-in KDA W/U counterfactual; never replace the public default" OFF)
option(ALA_PPU10_SOLVE_TF32X1_EXPERIMENT "Build an opt-in single-TF32 inverse; never replace the public default" OFF)
if(ALA_PPU10_SOLVE_TF32X1_EXPERIMENT AND NOT ALA_TARGET STREQUAL "ppu10")
  message(FATAL_ERROR "Single-TF32 experiment requires explicit ppu10")
endif()
if(ALA_PPU10_PARALLEL_WU_EXPERIMENT AND NOT ALA_TARGET STREQUAL "ppu10")
  message(FATAL_ERROR "Parallel W/U experiment requires explicit ppu10")
endif()
# PPU10保留已验证的token-parallel作为源代码构建默认；其他架构不迁移结果。
if(ALA_TARGET STREQUAL "ppu10")
  option(ALA_PREFILL_TOKEN_PARALLEL "PPU10 prefill token-parallel transport" ON)
else()
  option(ALA_PREFILL_TOKEN_PARALLEL "Experimental PPU10 prefill transport scheduling (device admission required)" OFF)
endif()
if(ALA_TARGET STREQUAL "ppu10")
  option(ALA_PPU10_BEST_CORE "按已验证shape启用最优已知core，其他shape保留基线" ON)
else()
  option(ALA_PPU10_BEST_CORE "按已验证shape启用最优已知PPU10 core" OFF)
endif()
if(ALA_PPU10_BEST_CORE AND NOT ALA_TARGET STREQUAL "ppu10")
  message(FATAL_ERROR "已验证最优core选路仅支持ppu10")
endif()
if(ALA_TARGET STREQUAL "ppu10" AND ALA_PREFILL_TOKEN_PARALLEL)
  option(ALA_PREFILL_DIRECT_PACK "H1: PPU10 prepare directly to packed BF16 layout" ON)
else()
  option(ALA_PREFILL_DIRECT_PACK "H1: PPU10 prepare directly to packed BF16 layout" OFF)
endif()
if(ALA_PREFILL_DIRECT_PACK AND (NOT ALA_TARGET STREQUAL "ppu10" OR NOT ALA_PREFILL_TOKEN_PARALLEL))
  message(FATAL_ERROR "Direct pack requires ppu10 and token-parallel ON")
endif()
if(ALA_PREFILL_TOKEN_PARALLEL AND NOT ALA_TARGET STREQUAL "ppu10")
  message(FATAL_ERROR "Token-parallel prefill experiment is scoped to ppu10")
endif()
set(_sources "${CMAKE_CURRENT_SOURCE_DIR}/gdn.cu" "${CMAKE_CURRENT_SOURCE_DIR}/prefill.cu"
             "${CMAKE_CURRENT_SOURCE_DIR}/qsa.cu" "${CMAKE_CURRENT_SOURCE_DIR}/conv_history.cu"
             "${CMAKE_CURRENT_SOURCE_DIR}/aux_pointwise.cu"
             "${CMAKE_CURRENT_SOURCE_DIR}/aux_shortconv.cu"
             "${CMAKE_CURRENT_SOURCE_DIR}/aux_hash.cu")

if(ALA_TARGET STREQUAL "cuda_sm120")
  enable_language(CUDA)
  add_library(actlize_la_serving SHARED ${_sources})
  set_target_properties(actlize_la_serving PROPERTIES CUDA_ARCHITECTURES 120 CUDA_STANDARD 17)
  target_compile_options(actlize_la_serving PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;-lineinfo>)
elseif(ALA_TARGET STREQUAL "ppu10" OR ALA_TARGET STREQUAL "ppu15")
  if(NOT PPU_SDK_ROOT)
    message(FATAL_ERROR "Set PPU_SDK_ROOT to the matching installed SDK")
  endif()
  if(ALA_TARGET STREQUAL "ppu15")
    set(_arch ppu0015)
    set(_numeric 150)
    list(APPEND _sources
      "${ALA_ROOT}/csrc/backends/ppu15/prepare.cu"
      "${ALA_ROOT}/csrc/backends/ppu15/state.cu"
      "${ALA_ROOT}/csrc/backends/ppu15/output.cu"
      "${ALA_ROOT}/csrc/backends/ppu15/launch.cu")
  else()
    set(_arch ppu0010)
    set(_numeric 100)
    # Minimal existing residual dependency closure, not all WY experiments.
    list(APPEND _sources
      "${ALA_ROOT}/csrc/backends/ppu10/launch.cu"
      "${ALA_ROOT}/csrc/gdn_chunk/gdn_wy_split_prepare_ppu.cu"
      "${ALA_ROOT}/csrc/gdn_chunk/gdn_wy_solve_static_ppu.cu"
      "${ALA_ROOT}/csrc/gdn_chunk/gdn_wy_residual_warps8_hvlayout_ppu.cu"
      "${ALA_ROOT}/csrc/gdn_chunk/gdn_wy_residual_gate_cache_ppu.cu"
      "${ALA_ROOT}/csrc/gdn_chunk/gdn_wy_residual_full_chunk_ppu.cu")
    if(ALA_PPU10_BEST_CORE)
      list(APPEND _sources "${ALA_ROOT}/csrc/gdn_chunk/gdn_wy_value64_ppu.cu")
    endif()
    if(ALA_PPU10_PARALLEL_WU_EXPERIMENT)
      list(APPEND _sources "${ALA_ROOT}/csrc/gdn_chunk/gdn_wy_parallel_wu_ppu.cu")
    endif()
    if(ALA_PPU10_SOLVE_TF32X1_EXPERIMENT)
      list(APPEND _sources "${ALA_ROOT}/csrc/gdn_chunk/gdn_wy_solve_tf32x1_ppu.cu")
    endif()
  endif()
  set(CUTLASS_PPU_ARCHS "${_arch}" CACHE STRING "PPU architecture" FORCE)
  foreach(_option EXAMPLES TOOLS LIBRARY TESTS GTEST_UNIT_TESTS)
    set(CUTLASS_ENABLE_${_option} OFF CACHE BOOL "" FORCE)
  endforeach()
  set(_actlize "${ALA_ROOT}/third_party/actlize")
  add_subdirectory("${_actlize}" actlize)
  foreach(_var CUTLASS_PPU_EXTRA_HGCC_FLAGS CUTLASS_PPU_DEV_INCLUDE_FLAGS PPU_DEVICE_HGCC_REAL)
    get_directory_property(${_var} DIRECTORY "${_actlize}" DEFINITION ${_var})
  endforeach()
  include("${ALA_ROOT}/cmake/GdnHgccArch.cmake")
  gdn_hgcc_target_flags("${_arch}" CUTLASS_PPU_EXTRA_HGCC_FLAGS _arch_contract
    "${PPU_DEVICE_HGCC_REAL}" ${CUTLASS_PPU_EXTRA_HGCC_FLAGS})
  # Preserve the native residual kernel's original math flags. Only the
  # recurrence/commit state seam uses explicit RN intrinsics to prevent an
  # accidental fused-vs-unfused record replay difference.
  list(APPEND CUTLASS_PPU_EXTRA_HGCC_FLAGS -std=c++17 -O3 "-DALA_PPU_TARGET=${_numeric}")
  if(ALA_TARGET STREQUAL "ppu10")
    list(APPEND CUTLASS_PPU_EXTRA_HGCC_FLAGS -DGDN_QSA_PPU=1)
    if(ALA_PPU10_BEST_CORE)
      list(APPEND CUTLASS_PPU_EXTRA_HGCC_FLAGS -DALA_PPU10_BEST_CORE=1)
    endif()
    if(ALA_PREFILL_TOKEN_PARALLEL)
      list(APPEND CUTLASS_PPU_EXTRA_HGCC_FLAGS -DALA_PREFILL_TOKEN_PARALLEL=1)
    endif()
    if(ALA_PREFILL_DIRECT_PACK)
      list(APPEND CUTLASS_PPU_EXTRA_HGCC_FLAGS -DALA_PREFILL_DIRECT_PACK=1)
    endif()
  endif()
  list(APPEND CUTLASS_PPU_DEV_INCLUDE_FLAGS "-I${ALA_ROOT}/include")
  cutlass_add_library(actlize_la_serving SHARED ${_sources})
  target_link_libraries(actlize_la_serving PRIVATE CUTLASS ppu::runtime)
  file(GLOB_RECURSE _headers CONFIGURE_DEPENDS
    "${ALA_ROOT}/include/*.hpp" "${ALA_ROOT}/include/*.cuh"
    "${ALA_ROOT}/csrc/serving/*.h" "${ALA_ROOT}/csrc/serving/*.cuh"
    "${ALA_ROOT}/csrc/backends/ppu15/*.h" "${ALA_ROOT}/csrc/backends/ppu15/*.cuh"
    "${ALA_ROOT}/csrc/backends/ppu10/*.h" "${ALA_ROOT}/csrc/backends/ppu10/*.cuh"
    "${ALA_ROOT}/csrc/gdn_chunk/*.cuh"
    "${_actlize}/include/*.h" "${_actlize}/include/*.hpp" "${_actlize}/include/*.inl")
  get_target_property(_objects actlize_la_serving SOURCES)
  foreach(_object IN LISTS _objects)
    if(_object MATCHES "[.]o$")
      add_custom_command(OUTPUT "${_object}" APPEND DEPENDS ${_headers} "${_arch_contract}")
    endif()
  endforeach()
else()
  message(FATAL_ERROR "ALA_TARGET must be ppu10, ppu15 or cuda_sm120; no architecture guessing")
endif()
target_link_options(actlize_la_serving PRIVATE -Wl,--no-undefined)
install(TARGETS actlize_la_serving LIBRARY DESTINATION lib)
install(FILES abi.h qsa.h aux_v1.h gdn_transaction_v2.h
        DESTINATION include/actlize_la/serving)

option(ALA_BUILD_SERVING_EXAMPLE "Build a Torch-free native graph consumer" ON)
if(ALA_BUILD_SERVING_EXAMPLE)
  add_executable(actlize_la_native_graph native_graph_example.cpp)
  target_link_libraries(actlize_la_native_graph PRIVATE actlize_la_serving)
  if(ALA_TARGET STREQUAL "cuda_sm120")
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(actlize_la_native_graph PRIVATE CUDA::cudart)
  else()
    target_compile_definitions(actlize_la_native_graph PRIVATE "ALA_PPU_TARGET=${_numeric}")
    target_include_directories(actlize_la_native_graph PRIVATE "${PPU_SDK_ROOT}/include")
    target_link_libraries(actlize_la_native_graph PRIVATE ppu::runtime)
  endif()
endif()
