# Independent cuLA-derived fused GDN forward

This is an **explicit candidate**, not a new default or a device-performance
admission. Legacy SM80/PPU1.0 kernels, reset/scan selection and tolerances stay
unchanged. PPU1.5 continues to belong to the actlize family; this work does not
claim its distinct device target has been admitted.

## Algorithm and attribution

Source: [cuLA C++ SM90 KDA](https://github.com/inclusionAI/cuLA/tree/79be249e61453808e18e5cef7702b363239e7d8d/csrc/kda/sm90),
revision `79be249e61453808e18e5cef7702b363239e7d8d`.
Only the needed forward/helper files are vendored under
`csrc/backends/sm90/cula`; original licenses/notices are retained.
`dev/backends/sm90_cula_sources.json` records the original per-file hashes.
The cuLA Python gate preprocessing, DSL kernels, SM100 and backward are not
dependencies. This is a scalar-GDN adaptation, not KDA relabelled as GDN.

The new path retains cuLA's important structure:

- One CTA per `(batch, value_head)`, native GVA head mapping, C64 tiles.
- Four warp groups: producer/store roles, two state/output math groups and
  one auxiliary QK/KK group; 512 threads per CTA.
- TMA Q/K/V input; BF16 WGMMA dense products; TF32 warp-MMA subchunk products;
  blocked inverse; FP32 resident state; asynchronous producer/consumer rings.
- One forward launch. No materialized global W/U or per-chunk state snapshots,
  gate-expansion kernel, device offsets kernel, or scratch-zeroing kernel.

The state dependency between chunks still exists. This is **not** a claim that
all chunks execute independently: different roles overlap independent work
inside the fused CTA. Grid is `B*Hv`, not `B*Hv*ceil(T/64)`. A small grid can
still underfill a device; neither CUTLASS nor fusion guarantees a win.

### Why scalar KDA equals GDN

Input `g[t,h]` is a natural-log decay increment, already processed by the model.
We do not apply a second softplus/A_log transform, normalize Q/K, or change beta.
With `p_i = sum(g_0..g_i)` **within this chunk**, define

```
A_ij = beta_i * dot(k_i,k_j) * exp(p_i-p_j), i>j; zero otherwise
R    = inverse(I+A) * diag(beta)
D    = R * (V - diag(exp(p))*K*H_in)
O    = (diag(exp(p))*Q*H_in + tril((Q*K^T)*exp(p_i-p_j))*D) / sqrt(128)
H_out= exp(p_last)*H_in + (diag(exp(p_last-p))*K)^T * D
```

This is the token recurrence `H <- exp(g)*H; delta=beta*(v-k*H);
H <- H+k^T*delta; o=q*H/sqrt(128)` in exact arithmetic. CPU FP64 tests compare
the two independently ordered formulas; **they do not prove device rounding,
async lifetime or speed**. BF16 intermediate rounding follows the fused cuLA
algorithm, not the legacy WY algorithm. The existing independent-recurrence
2% max/max criterion is unchanged; cross-algorithm RAW-BIT is not asserted.

## Adaptation seams

| Seam | Implemented adaptation / check |
|---|---|
| Scalar gate | Producer warp scans two 32-token segments, converts natural log to log2, restarts at each chunk. No expanded global gate. |
| Shared gate | Two 64-value FP32 buffers (512 B), CuTe zero-stride broadcast along K; preserve the consumer's nested K shape. Gate storage was 65,536 B in the direct vector-gate transplant. |
| State ABI | Public FP32 `[B,Hv,K,V]`, V-contiguous; cuLA's `(V,K)` accumulator uses a view permutation through `abi_layout.cuh`. No hidden transpose kernel. |
| Fixed batch | Host extents generate each sequence offset; no device `cu_seqlens` preparation. Arbitrary packed varlen/CP are not exposed. |
| Output tail | Predicated store for a nonfinal batch tail; it cannot overwrite the next sequence. No mutable per-SM tensor-map scratch. |
| Output lifetime | TMA store commit/wait and warp completion precede release of the output stage. Producer readiness and reader retirement are separate obligations. |
| CUTLASS API | PPU3.6 uses the two-argument accumulator-store selector and PipelineAsync constructor; CUDA uses its selected dependency. No copied legacy AIU pipeline. |

Whole dynamic shared storage is **167,936 B**, versus 232,448 B before scalar
gate compression, from the actual instantiated type. This is a footprint
fact, **not** measured occupancy or a speedup. Register/stack and static shared
usage must additionally be read from the compiled image and actual device.

The flat `(64,128,2)` gate layout initially failed the full-body compile:
cuLA's `flat_divide` consumer expects nested `(32,4)` in K. The final broadcast
preserves that hierarchy. Checking shape/element count alone would miss it.

## Public boundary

Scope: forward only; BF16 Q/K/V/beta; BF16 or FP32 g; Dk=Dv=128;
fixed positive B/T, native `Hv % Hk == 0`; optional FP32 initial/final state;
output-only supported. Input tensors must already be contiguous on one device.
No backward, FP16/FP8/FP4, QSA, ragged sequences, context parallelism or automatic
normalization is claimed. Initial numerical admission covers `g in [-1,0]`;
stronger gates/other distributions require separate finite/accuracy tests.

```python
from actlize_la import gdn_forward
# GDN_QSA_SM90_EXTENSION points to this target's built extension.
o, ht = gdn_forward(q, k, v, g, beta, initial_state=h0,
                   algorithm="fused_sm90", backend="ppu17")
```

Both algorithm and target are explicit. The extension exports its target and
math-contract identity. A CUDA or source-check binary cannot silently satisfy
a native PPU request. `implemented` in the catalog means source availability;
`device_admission=UNVERIFIED` is separate. Existing calls stay on their old routes.

## Build and one-call validation

Use a fresh directory under `/workspace`. No remote compilation is initiated by
these instructions; the user chooses the host/toolchain. An old SDK cannot be
made PPU1.7-capable by defining ACOMPUTE_VERSION. The builder first compiles a
real architecture receipt with the same flags as the kernel: compiler-provided
arch900 and SM90-all are mandatory. Native PPU additionally requires HGGC.

Native PPU1.7, **only with a matching toolchain**:

```bash
PPU_CUTLASS_ROOT=/path/to/ppu-cutlass3.6.0 \
PPU_SDK_ROOT=/path/to/ppu17-sdk \
python tools/build_gdn_sm90.py --target ppu17 \
  --out /workspace/gdn-sm90-ppu17-native
```

CUDA SM90a source/assembly and simulation-input build against the PPU fork:

```bash
python tools/build_gdn_sm90.py --target ppu17 --mode source-check \
  --compiler /usr/local/cuda/bin/nvcc --cutlass-root /root/cutlass3-3.6.0 \
  --out /workspace/gdn-sm90-ppu17-source
```

The second command produces **NVIDIA SM90a code**, not a PPU native binary.
For NVIDIA Hopper use `--target cuda_sm90` without the PPU dependency option.
CMake equivalents use `GDN_QSA_TARGET`, `GDN_SM90_COMPILER`, `GDN_SM90_MODE` and
`PPU_CUTLASS_ROOT`, with `GDN_QSA_BUILD_PPU_TESTS=OFF` for the new family.

One public call, no warmup/repeats/device oracle:

```bash
python tools/run_sm90_gdn.py --backend ppu17 \
  --extension /path/printed/by/build/_gdn_fused_sm90.cpython-312-x86_64-linux-gnu.so \
  --batch 2 --length 65 --q-heads 1 --v-heads 2 --initial --fp32-gate \
  --out /workspace/gdn-sm90-tail-check
```

Add `--source-check` only for the explicit source-check binary and actual
simulation-input workflow. No simulator command is invented here. Check the
simulator trace for exactly one fused kernel; a Python call count alone is not
trace evidence. The runner uses a CPU fixture/reference and CPU-side comparison,
records input/output/binary identities, and intentionally does not report
Python elapsed time as simulated performance.

After tail/GVA/initial-state checks, target case is
`--batch 1 --length 2048 --q-heads 16 --v-heads 32`, separately at `g=-0.1`
and `g=-1.0`. Keep the old algorithm as control. Do not infer a 1.5x result from
the reduced launch/workspace count; actual device or model measurements decide.

## Evidence limits

Local admission checks are documented in `SM90_FUSED_GDN_LOCAL_20260926.md`.
The subsequent physical-H800 checks are recorded separately in
`SM90_FUSED_GDN_H800_20260926.md`; those are Hopper control results, not
PPU1.7 native execution or a replacement for the PPU model.
The supplied SDK2.1.1 wrapper fails the real target gate (legacy architecture and
overridden ACOMPUTE selection); native PPU1.7 compilation is **SKIP: incapable
installed toolchain**, not PASS. Full native device correctness and performance
are **NOT RUN on PPU1.7**. No thresholds or old routing are relaxed to hide this gap.
