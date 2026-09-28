cmake_minimum_required(VERSION 3.19 FATAL_ERROR)

project(gdn_qsa_backends LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest)

option(GDN_QSA_ENABLE_PPU "Build the native PPU GDN backend" OFF)
include(cmake/GdnBackend.cmake)

set(
  PPU_SDK_ROOT
  "$ENV{PPU_SDK}"
  CACHE PATH "PPU SDK root used by PPU compile and ownership gates")
if(NOT PPU_SDK_ROOT AND
   EXISTS "/root/autodl-tmp/root-relocated/ppu-sdk/2.1.1")
  set(PPU_SDK_ROOT "/root/autodl-tmp/root-relocated/ppu-sdk/2.1.1")
endif()

add_library(gdn_qsa_backend_headers INTERFACE)
target_include_directories(
  gdn_qsa_backend_headers INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

option(GDN_QSA_BUILD_PPU_TESTS "Build legacy AIU host ownership tests" ${BUILD_TESTING})
if(GDN_QSA_BUILD_PPU_TESTS)
  include(cmake/backends/ppu_aiu_tests.cmake)
endif()

if(NOT GDN_QSA_RESOLVED_TARGET STREQUAL "none")
  include("${GDN_QSA_BACKEND_ENTRY}")
endif()
