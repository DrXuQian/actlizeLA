# actlize's logical target stays ppu0010. HGGC releases spell the compiler
# option either ppu_10 or ppu001; do not change architecture to make it compile.
function(gdn_hgcc_arch_flags output_flags output_contract compiler)
  set(_flags ${ARGN})
  set(_arch_flags ${_flags})
  list(FILTER _arch_flags INCLUDE REGEX "^(-arch|--gpu-architecture)=")
  list(LENGTH _arch_flags _arch_count)
  if(NOT _arch_count EQUAL 1 OR NOT "${_arch_flags}" MATCHES "^(-arch|--gpu-architecture)=(ppu_10|ppu001)$")
    message(FATAL_ERROR "GDN requires one logical PPU1.0 compiler target; got ${_arch_flags}")
  endif()
  set(_source "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../dev/ppu/hgcc_arch_probe.cu")
  # Unique outputs prevent an old .o from turning a no-output compiler into a
  # successful probe. No shared /tmp, deletion, version heuristic or device run.
  string(RANDOM LENGTH 12 ALPHABET 0123456789abcdef _nonce)
  set(_probe_dir "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/gdn-hgcc-arch-${_nonce}")
  file(MAKE_DIRECTORY "${_probe_dir}")
  foreach(_arch ppu_10 ppu001)
    set(_object "${_probe_dir}/${_arch}.o")
    execute_process(
      COMMAND "${compiler}" "-arch=${_arch}" -x hg -c "${_source}" -o "${_object}"
      RESULT_VARIABLE _rc OUTPUT_VARIABLE _stdout ERROR_VARIABLE _stderr
      TIMEOUT 45)
    set(_diagnostic "${_stdout}\n${_stderr}")
    file(WRITE "${_probe_dir}/${_arch}.log"
      "compiler=${compiler}\narchitecture=${_arch}\nreturncode=${_rc}\n${_diagnostic}")
    if("${_rc}" STREQUAL "0" AND EXISTS "${_object}")
      file(SIZE "${_object}" _size)
      if(_size GREATER 0)
        set(_selected "${_arch}")
        break()
      endif()
    endif()
    # Fall back ONLY for the known unsupported spelling. Missing SDK headers,
    # a broken compiler/loader or codegen failure is not an architecture alias.
    if(NOT "${_arch}" STREQUAL "ppu_10" OR "${_rc}" STREQUAL "0"
       OR NOT "${_diagnostic}" MATCHES "invalid value ['\"]ppu_10['\"] for option ['\"]gpu-architecture=")
      message(FATAL_ERROR
        "GDN HGGC PPU1.0 probe did not compile; no further architecture fallback allowed. See ${_probe_dir}/${_arch}.log\n${_diagnostic}")
    endif()
  endforeach()
  list(FILTER _flags EXCLUDE REGEX "^(-arch|--gpu-architecture)=")
  list(APPEND _flags "-arch=${_selected}")
  file(SHA256 "${compiler}" _compiler_sha)
  file(SHA256 "${_source}" _source_sha)
  set(_contract "${CMAKE_CURRENT_BINARY_DIR}/gdn_hgcc_arch.txt")
  # CONFIGURE preserves mtime when unchanged; SDK/alias changes invalidate all
  # device objects via this file's explicit build dependency.
  file(CONFIGURE OUTPUT "${_contract}" CONTENT
    "logical=ppu0010\ncompiler=${compiler}\ncompiler_sha256=${_compiler_sha}\nselected=-arch=${_selected}\nprobe_source_sha256=${_source_sha}\n" @ONLY)
  message(STATUS "GDN HGGC arch: logical=ppu0010 selected=-arch=${_selected} compile-probe=PASS (logs: ${_probe_dir})")
  set(${output_flags} ${_flags} PARENT_SCOPE)
  set(${output_contract} "${_contract}" PARENT_SCOPE)
endfunction()
