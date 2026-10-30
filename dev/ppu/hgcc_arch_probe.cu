// Compile only. Aliases must select the actual device generation, not just
// parse an empty TU. GDN_ARCH_EXPECTED is NOT the compiler's architecture macro.
#ifndef GDN_ARCH_EXPECTED
#error "architecture probe requires an explicit expected generation"
#endif
#if defined(__HGGC_ARCH__)
static_assert(__HGGC_ARCH__ == GDN_ARCH_EXPECTED, "HGGC selected another PPU generation");
#endif
extern "C" __global__ void gdn_hgcc_arch_probe(int* target) {
#if defined(__HGGC_ARCH__)
  *target = __HGGC_ARCH__;
#else
  (void)target;
#endif
}
