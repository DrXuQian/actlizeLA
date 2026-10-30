#pragma once
#include <cstdint>

// This host ABI contains no target headers. The opaque stream is the current
// PyTorch device stream, never the default stream or an independently created one.
extern "C" int actlize_la_ppu15_forward(
    void const* q, void const* k, void const* v, void const* g, void const* beta,
    float const* initial, void* output, float* final, void* inverse,
    void* snapshots, void* vnew, float* gates,
    int batch, int sequence, int q_heads, int value_heads, bool gate_fp32, void* stream);
extern "C" int actlize_la_ppu15_target_arch();
