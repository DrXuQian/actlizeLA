cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

project(actlizeLA LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ACTLIZELA_ENABLE_PPU "Build the PPU forward library and device ABI test" OFF)
option(ACTLIZELA_ENABLE_LOCAL_GATES "Register NVCC proof runners with CTest" OFF)

include(GNUInstallDirs)
include(CTest)

add_library(actlize_la_headers INTERFACE)
target_include_directories(
  actlize_la_headers
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

add_executable(chunked_gdn_oracle dev/gates/l203_chunked_gdn_oracle.cpp)
target_link_libraries(chunked_gdn_oracle PRIVATE actlize_la_headers)

if(BUILD_TESTING)
  add_test(NAME chunked_gdn_oracle COMMAND chunked_gdn_oracle)
  if(ACTLIZELA_ENABLE_LOCAL_GATES)
    add_test(
      NAME chunked_gdn_local_gates
      COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tools/run_local_gates.sh)
  endif()
endif()

if(ACTLIZELA_ENABLE_PPU)
  set(CUTLASS_PPU_ARCHS ppu0010 CACHE STRING "PPU architectures" FORCE)
  set(CUTLASS_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(CUTLASS_ENABLE_TOOLS OFF CACHE BOOL "" FORCE)
  set(CUTLASS_ENABLE_LIBRARY OFF CACHE BOOL "" FORCE)
  set(CUTLASS_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
  set(CUTLASS_ENABLE_GTEST_UNIT_TESTS OFF CACHE BOOL "" FORCE)

  add_subdirectory(third_party/actlize actlize)

  # actlize compiles .cu sources with an hgcc custom command. Target include
  # properties and normal variables do not cross back out of add_subdirectory,
  # so bridge the subdirectory's exact compiler/flags and add this repository's
  # public include directory explicitly.
  get_directory_property(
    _actlize_hgcc_flags
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/third_party/actlize
    DEFINITION CUTLASS_PPU_EXTRA_HGCC_FLAGS)
  get_directory_property(
    _actlize_device_includes
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/third_party/actlize
    DEFINITION CUTLASS_PPU_DEV_INCLUDE_FLAGS)
  get_directory_property(
    _actlize_device_hgcc
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/third_party/actlize
    DEFINITION PPU_DEVICE_HGCC_REAL)
  if(NOT _actlize_device_hgcc OR NOT EXISTS "${_actlize_device_hgcc}")
    message(FATAL_ERROR
      "actlize's PPU device compiler did not cross the subdirectory boundary: "
      "PPU_DEVICE_HGCC_REAL='${_actlize_device_hgcc}'")
  endif()
  set(PPU_DEVICE_HGCC_REAL ${_actlize_device_hgcc})
  set(CUTLASS_PPU_EXTRA_HGCC_FLAGS ${_actlize_hgcc_flags})
  set(CUTLASS_PPU_DEV_INCLUDE_FLAGS ${_actlize_device_includes})
  list(APPEND CUTLASS_PPU_DEV_INCLUDE_FLAGS
       "-I${CMAKE_CURRENT_SOURCE_DIR}/include")

  cutlass_add_library(
    actlize_la_ppu SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ppu_chunked_gdn_backend.cu)
  target_include_directories(
    actlize_la_ppu
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
  target_link_libraries(actlize_la_ppu PRIVATE CUTLASS ppu::runtime)
  set_target_properties(actlize_la_ppu PROPERTIES OUTPUT_NAME actlize_la_ppu)

  add_executable(test_ppu_chunked_gdn_abi tests/test_ppu_chunked_gdn_abi.cpp)
  target_compile_definitions(test_ppu_chunked_gdn_abi PRIVATE SWITCH_TO_HGGCRT)
  target_link_libraries(
    test_ppu_chunked_gdn_abi PRIVATE actlize_la_ppu ppu::runtime)
  if(BUILD_TESTING)
    add_test(NAME ppu_chunked_gdn_abi COMMAND test_ppu_chunked_gdn_abi)
  endif()

  install(
    TARGETS actlize_la_ppu
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
