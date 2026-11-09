#pragma once

// Retained PPU1.0 residual kernel family. Configure outside capture; launch
// consumes caller-owned buffers and uses the caller's current stream.
extern "C" int actlize_la_ppu10_configure();
// Host-only查询：0保留、1 solve8、2 solve8+V64/256；不读取device内容。
extern "C" int actlize_la_ppu10_core_route(int batch, int sequence, int key_heads, int value_heads);
extern "C" int actlize_la_ppu10_forward_prepared(
    void const* q, void const* k, void const* v, void const* g, void const* beta,
    float const* initial, void* output, float* final, void* inverse,
    void* snapshots, void* vnew, float* gates,
    int batch, int sequence, int q_heads, int value_heads, bool gate_fp32, void* stream);
