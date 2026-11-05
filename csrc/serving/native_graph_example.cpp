// Standalone native consumer: no Python/Torch, no runtime compilation.
#include "aux_v1.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#ifdef ALA_PPU_TARGET
#include <hggc_runtime_api.h>
#define RT(x) hggc##x
#else
#include <cuda_runtime_api.h>
#define RT(x) cuda##x
#endif
#define CHECK(call)                                                            \
  do {                                                                         \
    auto e = (call);                                                           \
    if (int(e)) {                                                              \
      std::fprintf(stderr, "%s: status=%d\n", #call, int(e));                  \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static uint16_t bf(float x) {
  uint32_t b;
  std::memcpy(&b, &x, 4);
  return uint16_t((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}
static float fp(uint16_t x) {
  uint32_t b = uint32_t(x) << 16;
  float f;
  std::memcpy(&f, &b, 4);
  return f;
}
static ala_tensor_v1 tensor(void *ptr, int dtype, int64_t a, int64_t b = 0,
                            int64_t c = 0) {
  ala_tensor_v1 t{};
  t.data = ptr;
  t.dtype = dtype;
  t.rank = c ? 3 : b ? 2 : 1;
  t.shape[0] = a;
  t.shape[1] = b;
  t.shape[2] = c;
  int64_t n = 1;
  for (int d = t.rank - 1; d >= 0; --d) {
    t.stride[d] = n;
    n *= t.shape[d];
  }
  t.nbytes = n * (dtype == ALA_BF16 ? 2 : 4);
  return t;
}
int main(int argc, char **argv) {
  if (argc == 2 && !std::strcmp(argv[1], "--describe")) {
    std::printf("native-graph-example auxiliary-v1 struct_bytes=%zu; no "
                "Torch/Python callback\n",
                sizeof(ala_pointwise_v1));
    return 0;
  }
  constexpr int T = 8, G = 4, D = 2560;
  std::vector<uint16_t> hx(T * G * D), hy(T * G * D);
  std::vector<float> hw(G * D);
  for (int i = 0; i < T * G * D; ++i)
    hx[i] = bf(float(i % 29 - 14) * 0.125f);
  for (int i = 0; i < G * D; ++i)
    hw[i] = 1.0f + float(i % 7) * 0.03125f;
  void *x = nullptr, *y = nullptr, *w = nullptr, *live = nullptr,
       *errors = nullptr;
  CHECK(RT(Malloc)(&x, hx.size() * 2));
  CHECK(RT(Malloc)(&y, hy.size() * 2));
  CHECK(RT(Malloc)(&w, hw.size() * 4));
  CHECK(RT(Malloc)(&live, 4));
  CHECK(RT(Malloc)(&errors, T * 4));
  RT(Stream_t) stream;
  CHECK(RT(StreamCreate)(&stream));
  CHECK(RT(MemcpyAsync)(x, hx.data(), hx.size() * 2, RT(MemcpyHostToDevice),
                        stream));
  CHECK(RT(MemcpyAsync)(w, hw.data(), hw.size() * 4, RT(MemcpyHostToDevice),
                        stream));
  int count = T;
  CHECK(RT(MemcpyAsync)(live, &count, 4, RT(MemcpyHostToDevice), stream));
  ala_pointwise_v1 p{};
  p.header = {1, sizeof(p)};
  p.rows.tokens = T;
  p.rows.num_tokens = tensor(live, ALA_I32, 1);
  p.kind = ALA_GROUPED_NORM;
  p.groups = G;
  p.width = D;
  p.epsilon = 1e-6f;
  p.weight_per_group = 1;
  p.x = tensor(x, ALA_BF16, T, G, D);
  p.weight = tensor(w, ALA_F32, G, D);
  p.output = tensor(y, ALA_BF16, T, G, D);
  p.errors = tensor(errors, ALA_I32, T);
  CHECK(ala_aux_v1_validate(2, &p));
  CHECK(ala_pointwise_v1_run(&p, stream));
  CHECK(RT(StreamSynchronize)(stream)); // preparation only
  RT(Graph_t) graph;
  RT(GraphExec_t) executable;
  CHECK(RT(StreamBeginCapture)(stream, RT(StreamCaptureModeGlobal)));
  CHECK(ala_pointwise_v1_run(&p, stream));
  CHECK(RT(StreamEndCapture)(stream, &graph));
  CHECK(RT(GraphInstantiateWithFlags)(&executable, graph, 0));
  size_t nodes = 0;
  CHECK(RT(GraphGetNodes)(graph, nullptr, &nodes));
  if (nodes != 1) {
    std::fprintf(stderr, "expected one native kernel, got %zu nodes\n", nodes);
    return 1;
  }
  for (count = 1; count <= T; ++count) {
    CHECK(RT(MemcpyAsync)(live, &count, 4, RT(MemcpyHostToDevice), stream));
    // This is the entire serving operation. Rust can issue the same call.
    CHECK(RT(GraphLaunch)(executable, stream));
    // Verification below is a harness observation, never part of the graph.
    CHECK(RT(MemcpyAsync)(hy.data(), y, hy.size() * 2, RT(MemcpyDeviceToHost),
                          stream));
    int status[T];
    CHECK(RT(MemcpyAsync)(status, errors, sizeof(status),
                          RT(MemcpyDeviceToHost), stream));
    CHECK(RT(StreamSynchronize)(stream));
    for (int t = 0; t < T; ++t)
      for (int g = 0; g < G; ++g) {
        if (status[t])
          return 1;
        float square = 0;
        for (int d = 0; d < D; ++d) {
          float z = fp(hx[(t * G + g) * D + d]);
          square += z * z;
        }
        float inv = 1.0f / std::sqrt(square / D + 1e-6f);
        for (int d = 0; d < D; ++d) {
          int i = (t * G + g) * D + d;
          uint16_t expected =
              t < count ? bf(fp(hx[i]) * inv * hw[g * D + d]) : 0;
          if (std::abs(int(expected) - int(hy[i])) > 1)
            return 1;
        }
      }
  }
  CHECK(RT(GraphExecDestroy)(executable));
  CHECK(RT(GraphDestroy)(graph));
  CHECK(RT(StreamDestroy)(stream));
  CHECK(RT(Free)(x));
  CHECK(RT(Free)(y));
  CHECK(RT(Free)(w));
  CHECK(RT(Free)(live));
  CHECK(RT(Free)(errors));
  std::puts("[FlashNext native] PASS replays=8 fixed-addresses kernel_nodes=1 "
            "no-Torch no-Python no-runtime-JIT");
}
