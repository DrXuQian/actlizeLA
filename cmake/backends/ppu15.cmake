# Native PPU1.5 graph: no legacy PPU1.0 kernel, no Hopper dependency.
if(NOT GDN_QSA_RESOLVED_TARGET STREQUAL "ppu15")
  message(FATAL_ERROR "PPU1.5 source graph rejects another target")
endif()
if(NOT PPU_SDK_ROOT)
  message(FATAL_ERROR "PPU1.5 requires PPU_SDK_ROOT or PPU_SDK")
endif()
set(CUTLASS_PPU_ARCHS ppu0015 CACHE STRING "PPU architectures" FORCE)
foreach(_option EXAMPLES TOOLS LIBRARY TESTS GTEST_UNIT_TESTS)
  set(CUTLASS_ENABLE_${_option} OFF CACHE BOOL "" FORCE)
endforeach()
set(PPU_SDK_ROOT "${PPU_SDK_ROOT}" CACHE PATH "" FORCE)
add_subdirectory("${GDN_QSA_DEPENDENCY_ROOT}" actlize)
foreach(_var CUTLASS_PPU_EXTRA_HGCC_FLAGS CUTLASS_PPU_DEV_INCLUDE_FLAGS PPU_DEVICE_HGCC_REAL)
  get_directory_property(${_var} DIRECTORY "${GDN_QSA_DEPENDENCY_ROOT}" DEFINITION ${_var})
endforeach()
include(cmake/GdnHgccArch.cmake)
gdn_hgcc_target_flags(ppu0015 CUTLASS_PPU_EXTRA_HGCC_FLAGS _gdn_arch_contract
  "${PPU_DEVICE_HGCC_REAL}" ${CUTLASS_PPU_EXTRA_HGCC_FLAGS})
list(APPEND CUTLASS_PPU_EXTRA_HGCC_FLAGS -std=c++17 -O3)
list(APPEND CUTLASS_PPU_DEV_INCLUDE_FLAGS "-I${CMAKE_CURRENT_SOURCE_DIR}/include")

# Explicit opt-in TU list. New experiments cannot enter the shipping graph by
# being added to a directory. Native tests are a separate opt-in target below.
cutlass_add_library(actlize_la_ppu15 SHARED
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/backends/ppu15/prepare.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/backends/ppu15/state.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/backends/ppu15/output.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/backends/ppu15/launch.cu)
target_link_libraries(actlize_la_ppu15 PRIVATE CUTLASS ppu::runtime)
target_link_options(actlize_la_ppu15 PRIVATE -Wl,--no-undefined)

# hgcc custom objects have no transitive depfile. Header changes, including
# the pinned native atoms, must invalidate them, not merely the binding DSO.
file(GLOB_RECURSE _ppu15_headers CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/backends/ppu15/*.cuh"
  "${CMAKE_CURRENT_SOURCE_DIR}/csrc/backends/ppu15/*.h"
  "${GDN_QSA_DEPENDENCY_ROOT}/include/*.h"
  "${GDN_QSA_DEPENDENCY_ROOT}/include/*.hpp"
  "${GDN_QSA_DEPENDENCY_ROOT}/include/*.inl")
list(APPEND _ppu15_headers "${_gdn_arch_contract}"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/gdn_qsa/wy_contract.hpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/include/gdn_qsa/ppu/wy_solve_static.cuh")
get_target_property(_objects actlize_la_ppu15 SOURCES)
foreach(_object IN LISTS _objects)
  if(_object MATCHES "[.]o$")
    add_custom_command(OUTPUT "${_object}" APPEND DEPENDS ${_ppu15_headers})
  endif()
endforeach()

# Host binding uses the installed PPU PyTorch headers/libraries. Do not invoke
# NVIDIA CUDA discovery or manufacture a libhggcrt.12 -> .13 compatibility link.
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(COMMAND "${Python3_EXECUTABLE}" -c
  "import json,torch; from torch.utils.cpp_extension import include_paths,library_paths; print(json.dumps({'includes':include_paths(),'libraries':library_paths(),'abi':int(torch._C._GLIBCXX_USE_CXX11_ABI)}))"
  OUTPUT_VARIABLE _torch_json OUTPUT_STRIP_TRAILING_WHITESPACE RESULT_VARIABLE _torch_status)
if(NOT _torch_status EQUAL 0)
  message(FATAL_ERROR "PPU1.5 binding requires PyTorch from the matching PPU SDK container")
endif()
string(JSON _torch_abi GET "${_torch_json}" abi)
foreach(_kind includes libraries)
  string(JSON _count LENGTH "${_torch_json}" ${_kind})
  math(EXPR _last "${_count}-1")
  foreach(_index RANGE 0 ${_last})
    string(JSON _entry GET "${_torch_json}" ${_kind} ${_index})
    list(APPEND _torch_${_kind} "${_entry}")
  endforeach()
endforeach()
Python3_add_library(_gdn_residual_ppu15 MODULE WITH_SOABI csrc/backends/ppu15/bindings.cpp)
target_compile_definitions(_gdn_residual_ppu15 PRIVATE TORCH_EXTENSION_NAME=_gdn_residual_ppu15
  _GLIBCXX_USE_CXX11_ABI=${_torch_abi})
target_include_directories(_gdn_residual_ppu15 PRIVATE ${_torch_includes}
  "${PPU_SDK_ROOT}/include" "${PPU_SDK_ROOT}/targets/x86_64-linux/include"
  "${PPU_SDK_ROOT}/CUDA_SDK/targets/x86_64-linux/include"
  "${PPU_SDK_ROOT}/CUDA_SDK/targets/x86_64-linux/include/cccl")
target_link_directories(_gdn_residual_ppu15 PRIVATE ${_torch_libraries})
target_link_libraries(_gdn_residual_ppu15 PRIVATE actlize_la_ppu15 ppu::runtime
  torch_python torch torch_cpu torch_cuda c10 c10_cuda)
set_target_properties(_gdn_residual_ppu15 PROPERTIES BUILD_RPATH "${_torch_libraries};$ORIGIN" INSTALL_RPATH "$ORIGIN")

option(GDN_QSA_PPU15_DEVICE_TESTS "Build independent native PPU1.5 primitive checks" OFF)
if(GDN_QSA_PPU15_DEVICE_TESTS)
  cutlass_add_executable(test_ppu15_primitives ${CMAKE_CURRENT_SOURCE_DIR}/tests/ppu15_device.cu)
  target_link_libraries(test_ppu15_primitives PRIVATE CUTLASS ppu::runtime)
  get_target_property(_test_objects test_ppu15_primitives SOURCES)
  foreach(_object IN LISTS _test_objects)
    if(_object MATCHES "[.]o$")
      add_custom_command(OUTPUT "${_object}" APPEND DEPENDS ${_ppu15_headers})
    endif()
  endforeach()
endif()
