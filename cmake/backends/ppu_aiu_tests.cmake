# Host ownership/layout checks for the legacy PPU AIU implementation only.
add_executable(l004_ppu_affine_pipeline dev/ppu/l004_affine_pipeline.cpp)
add_executable(l006_ppu_original_delivery dev/ppu/l006_original_kernel_delivery.cpp)
target_include_directories(l006_ppu_original_delivery PRIVATE
  ${PPU_SDK_ROOT}/include
  ${PPU_SDK_ROOT}/CUDA_SDK/targets/x86_64-linux/include
  ${PPU_SDK_ROOT}/CUDA_SDK/targets/x86_64-linux/include/cccl
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/actlize/include)
target_link_libraries(l006_ppu_original_delivery PRIVATE gdn_qsa_backend_headers)
add_executable(l007_ppu_wy_ownership dev/ppu/l007_wy_ownership.cpp)
target_include_directories(l007_ppu_wy_ownership PRIVATE
  ${PPU_SDK_ROOT}/include ${PPU_SDK_ROOT}/CUDA_SDK/targets/x86_64-linux/include
  ${PPU_SDK_ROOT}/CUDA_SDK/targets/x86_64-linux/include/cccl
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/actlize/include)
target_link_libraries(l007_ppu_wy_ownership PRIVATE gdn_qsa_backend_headers)
add_executable(l008_ppu_wy_delivery dev/ppu/l008_wy_delivery.cpp)
target_include_directories(l008_ppu_wy_delivery PRIVATE
  ${PPU_SDK_ROOT}/include ${PPU_SDK_ROOT}/CUDA_SDK/targets/x86_64-linux/include
  ${PPU_SDK_ROOT}/CUDA_SDK/targets/x86_64-linux/include/cccl
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/actlize/include)
target_link_libraries(l008_ppu_wy_delivery PRIVATE gdn_qsa_backend_headers)
add_executable(l009_ppu_wy_tiles dev/ppu/l009_wy_tiles.cpp)
target_include_directories(l009_ppu_wy_tiles PRIVATE
  ${PPU_SDK_ROOT}/include ${PPU_SDK_ROOT}/CUDA_SDK/targets/x86_64-linux/include
  ${PPU_SDK_ROOT}/CUDA_SDK/targets/x86_64-linux/include/cccl
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/actlize/include)
target_link_libraries(l009_ppu_wy_tiles PRIVATE gdn_qsa_backend_headers)
add_executable(l010_ppu_wy_state_address dev/ppu/l010_wy_state_address.cpp)
target_include_directories(l010_ppu_wy_state_address PRIVATE
  ${PPU_SDK_ROOT}/include ${PPU_SDK_ROOT}/CUDA_SDK/targets/x86_64-linux/include
  ${PPU_SDK_ROOT}/CUDA_SDK/targets/x86_64-linux/include/cccl
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/actlize/include)
target_link_libraries(l010_ppu_wy_state_address PRIVATE gdn_qsa_backend_headers)
add_executable(l011_ppu_wy_stage_address dev/ppu/l011_wy_stage_address.cpp)
target_include_directories(l011_ppu_wy_stage_address PRIVATE
  ${PPU_SDK_ROOT}/include ${PPU_SDK_ROOT}/CUDA_SDK/targets/x86_64-linux/include
  ${PPU_SDK_ROOT}/CUDA_SDK/targets/x86_64-linux/include/cccl
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/actlize/include)
target_link_libraries(l011_ppu_wy_stage_address PRIVATE gdn_qsa_backend_headers)
add_executable(l012_ppu_wy_prepare_rows dev/ppu/l012_wy_prepare_rows.cpp)
target_include_directories(l012_ppu_wy_prepare_rows PRIVATE
  ${PPU_SDK_ROOT}/include ${PPU_SDK_ROOT}/CUDA_SDK/targets/x86_64-linux/include
  ${PPU_SDK_ROOT}/CUDA_SDK/targets/x86_64-linux/include/cccl
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/actlize/include)
target_link_libraries(l012_ppu_wy_prepare_rows PRIVATE gdn_qsa_backend_headers)
add_executable(l016_ppu_wy_aiu_pair dev/ppu/l016_wy_aiu_pair.cpp)
target_include_directories(l016_ppu_wy_aiu_pair PRIVATE
  ${PPU_SDK_ROOT}/include ${PPU_SDK_ROOT}/CUDA_SDK/targets/x86_64-linux/include
  ${PPU_SDK_ROOT}/CUDA_SDK/targets/x86_64-linux/include/cccl
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/actlize/include)
target_link_libraries(l016_ppu_wy_aiu_pair PRIVATE gdn_qsa_backend_headers)
add_executable(l018_ppu_wy_split_prepare dev/ppu/l018_wy_split_prepare.cpp)
target_include_directories(l018_ppu_wy_split_prepare PRIVATE
  ${PPU_SDK_ROOT}/include ${PPU_SDK_ROOT}/CUDA_SDK/targets/x86_64-linux/include
  ${PPU_SDK_ROOT}/CUDA_SDK/targets/x86_64-linux/include/cccl
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/actlize/include)
target_link_libraries(l018_ppu_wy_split_prepare PRIVATE gdn_qsa_backend_headers)
add_executable(l019_ppu_wy_state_pipeline dev/ppu/l019_wy_state_pipeline.cpp)
target_include_directories(l019_ppu_wy_state_pipeline PRIVATE
  ${PPU_SDK_ROOT}/include ${PPU_SDK_ROOT}/CUDA_SDK/targets/x86_64-linux/include
  ${PPU_SDK_ROOT}/CUDA_SDK/targets/x86_64-linux/include/cccl
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/actlize/include)
target_link_libraries(l019_ppu_wy_state_pipeline PRIVATE gdn_qsa_backend_headers)
add_executable(l020_ppu_wy_residual dev/ppu/l020_wy_residual.cpp)
target_include_directories(l020_ppu_wy_residual PRIVATE
  ${PPU_SDK_ROOT}/include ${PPU_SDK_ROOT}/CUDA_SDK/targets/x86_64-linux/include
  ${PPU_SDK_ROOT}/CUDA_SDK/targets/x86_64-linux/include/cccl
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/actlize/include)
target_link_libraries(l020_ppu_wy_residual PRIVATE gdn_qsa_backend_headers)
add_executable(l021_ppu_wy_residual_prefetch dev/ppu/l021_wy_residual_prefetch.cpp)
target_include_directories(l021_ppu_wy_residual_prefetch PRIVATE
  ${PPU_SDK_ROOT}/include ${PPU_SDK_ROOT}/CUDA_SDK/targets/x86_64-linux/include
  ${PPU_SDK_ROOT}/CUDA_SDK/targets/x86_64-linux/include/cccl
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/actlize/include)
target_link_libraries(l021_ppu_wy_residual_prefetch PRIVATE gdn_qsa_backend_headers)
foreach(_check IN ITEMS l022_wy_residual_v16 l023_wy_residual_operands l024_wy_residual_blayout l025_wy_residual_warps8 l027_wy_residual_warps8_blayout l028_wy_residual_warps8_operands l029_wy_residual_warps8_hlayout l030_wy_residual_warps8_hvlayout l031_wy_residual_metadata l032_wy_solve_static l033_wy_gate_cache)
  add_executable(${_check} dev/ppu/${_check}.cpp)
  target_include_directories(${_check} PRIVATE ${PPU_SDK_ROOT}/include
    ${PPU_SDK_ROOT}/CUDA_SDK/targets/x86_64-linux/include
    ${PPU_SDK_ROOT}/CUDA_SDK/targets/x86_64-linux/include/cccl
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/actlize/include)
  target_link_libraries(${_check} PRIVATE gdn_qsa_backend_headers)
  if(BUILD_TESTING)
    add_test(NAME ${_check} COMMAND ${_check})
  endif()
endforeach()

if(BUILD_TESTING)
  add_test(NAME ppu_affine_pipeline COMMAND l004_ppu_affine_pipeline)
  add_test(NAME ppu_original_delivery COMMAND l006_ppu_original_delivery)
  add_test(NAME ppu_wy_ownership COMMAND l007_ppu_wy_ownership)
  add_test(NAME ppu_wy_delivery COMMAND l008_ppu_wy_delivery)
  add_test(NAME ppu_wy_tiles COMMAND l009_ppu_wy_tiles)
  add_test(NAME ppu_wy_state_address COMMAND l010_ppu_wy_state_address)
  add_test(NAME ppu_wy_stage_address COMMAND l011_ppu_wy_stage_address)
  add_test(NAME ppu_wy_prepare_rows COMMAND l012_ppu_wy_prepare_rows)
  add_test(NAME ppu_wy_aiu_pair COMMAND l016_ppu_wy_aiu_pair)
  add_test(NAME ppu_wy_split_prepare COMMAND l018_ppu_wy_split_prepare)
  add_test(NAME ppu_wy_state_pipeline COMMAND l019_ppu_wy_state_pipeline)
  add_test(NAME ppu_wy_residual COMMAND l020_ppu_wy_residual)
  add_test(NAME ppu_wy_residual_prefetch COMMAND l021_ppu_wy_residual_prefetch)
endif()
