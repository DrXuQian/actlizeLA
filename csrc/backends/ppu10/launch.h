#pragma once

// Retained PPU1.0 residual kernel family. Configure outside capture; launch
// consumes caller-owned buffers and uses the caller's current stream.
extern "C" int actlize_la_ppu10_configure();
// Preparation-time host metadata query; 0=retained, 3=native BF16.
// A distinct v2 symbol avoids changing development libraries' four-int ABI.
extern "C" int actlize_la_ppu10_core_route_v2(
    int batch, int sequence, int key_heads, int value_heads, bool gate_fp32);
extern "C" int actlize_la_ppu10_forward_prepared(
    void const* q, void const* k, void const* v, void const* g, void const* beta,
    float const* initial, void* output, float* final, void* inverse,
    void* snapshots, void* vnew, float* gates,
    int batch, int sequence, int q_heads, int value_heads, bool gate_fp32, void* stream);
