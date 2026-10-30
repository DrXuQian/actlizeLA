add_executable(ppu15_layout tests/ppu15_layout.cpp)
target_include_directories(ppu15_layout PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/third_party/actlize/include
  ${PPU_SDK_ROOT}/include
  ${PPU_SDK_ROOT}/CUDA_SDK/targets/x86_64-linux/include
  ${PPU_SDK_ROOT}/CUDA_SDK/targets/x86_64-linux/include/cccl)
if(BUILD_TESTING)
  add_test(NAME ppu15_layout COMMAND ppu15_layout)
  add_test(NAME ppu15_old_result_negative COMMAND ppu15_layout --plant-old-c)
  set_tests_properties(ppu15_old_result_negative PROPERTIES WILL_FAIL TRUE)
endif()
