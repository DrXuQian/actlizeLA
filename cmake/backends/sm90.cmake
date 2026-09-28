# Complete new algorithm. No actlize add_subdirectory or SM80 objects.
if(NOT GDN_QSA_RESOLVED_TARGET MATCHES "^(cuda_sm90|ppu17)$")
  message(FATAL_ERROR "SM90 source graph rejects ${GDN_QSA_RESOLVED_TARGET}")
endif()
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(GDN_SM90_MODE native CACHE STRING "native or source-check (not PPU device code)")
set(GDN_SM90_CONFIGURATION control CACHE STRING "control, value64, value64-local-inverse, or value128-paired")
set(GDN_SM90_CUTLASS_ROOT "" CACHE PATH "Explicit CUDA CUTLASS dependency (ignored on PPU17)")
set(GDN_SM90_COMPILER "" CACHE FILEPATH "Explicit SM90a-capable compiler")
set(PPU_CUTLASS_ROOT "$ENV{PPU_CUTLASS_ROOT}" CACHE PATH "PPU CUTLASS3.6/10700")
if(GDN_QSA_RESOLVED_TARGET STREQUAL "ppu17" AND NOT PPU_CUTLASS_ROOT)
  message(FATAL_ERROR "ppu17 requires PPU_CUTLASS_ROOT; not an actlize target")
endif()
set(_sm90_args --target ${GDN_QSA_RESOLVED_TARGET} --mode ${GDN_SM90_MODE}
  --configuration ${GDN_SM90_CONFIGURATION}
  --out ${CMAKE_CURRENT_BINARY_DIR}/sm90)
if(GDN_SM90_COMPILER)
  list(APPEND _sm90_args --compiler ${GDN_SM90_COMPILER})
endif()
if(GDN_QSA_RESOLVED_TARGET STREQUAL "ppu17")
  list(APPEND _sm90_args --cutlass-root ${PPU_CUTLASS_ROOT})
elseif(GDN_SM90_CUTLASS_ROOT)
  list(APPEND _sm90_args --cutlass-root ${GDN_SM90_CUTLASS_ROOT})
endif()
add_custom_target(gdn_fused_sm90 ALL
  COMMAND ${CMAKE_COMMAND} -E env "PPU_SDK_ROOT=${PPU_SDK_ROOT}"
    ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/build_gdn_sm90.py ${_sm90_args}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} USES_TERMINAL)
