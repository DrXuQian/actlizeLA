# Residual B-oriented native shared layout

Parent b2e8f00; measured control source5a3ebc9. Opt-in `residual-blayout`;
default routing, admitted residual, V16 and operand-prefetch arms are unchanged.
Priority workload B1/S2048/Hk16/Hv32/K=V128/C64, V32/128threads/grid128.
Both g=-0.1 and-1.0 need numeric admission; default capture is-1.0.

Device follow-up on3c7da09: RAW-BIT passes, BC falls22.70%, but state
130.484->130.334us and full ACU sum230.678->232.220us show no latency gain.
Keep opt-in, no promotion. See [verified capture and interpretation](PPU_GDN_RESIDUAL_B_LAYOUT_ACU_20260924.md).
The local handoff below remains historical compile evidence, not the verdict.

## Change and boundary

Only `residual` and `scaled-V` change their physical shared placement.
Their existing BF16 stores directly write native16x16 cubes in B(value,time)
orientation. The paired non-transposed PPU `ldmatrix.swzl` yields the native
MMA B fragment without register exchange. There is no transpose kernel,
additional shared buffer, copy, barrier, or shuffle. The CuTe composed layout
is checked against real native C/B traits and the actual actlize load simulator.
A precomputed lane base plus immediate C-slot offsets avoids a general
per-element transpose/address expression in the recurrence.

K/P/V global AIU.swzl inputs, H snapshot, Vnew publication and final FP32
state remain untouched. The same algorithm, K order, BF16 boundaries, CTA
ownership and global ABI remain in force. In particular H has a second,
vector-publication consumer, so it is not transposed by this shortcut.
See [all-stage reference audit](PPU_GDN_AIU_DELIVERY_AUDIT.md).

This targets the internal B reads, **not every shared access in GDN**. Scalar
BF16 stores, H snapshot and K transpose loads remain. The host bank model is
explicitly32 four-byte banks and one32-bit word/lane phase: every one of the32
modeled phases visits32 distinct words/banks. Hardware request grouping and
whole-kernel BC are not proven by that model; ACU must report read/write BC
separately, normalized by requests and useful work. Do not claim BC=0 yet.

## Native compile evidence, not a performance result

SDK2.1.1, native PPU0010; full device bodies linked:

| Property | residual control | B-layout candidate |
|---|---:|---:|
| Registers / stack B | 242 /0 | 242 /0 |
| Shared B / CTA threads | 45568 /128 | 45568 /128 |
| AIU / BF16 MMA / barrier static sites | 4 /40 /5 | 4 /40 /5 |
| Non-transposed / transposed SWZL sites | 24 /32 | 32 /24 |
| Whole-body static sites | 2092 | 2093 |
| Recurrence backedge-body sites (two exits) | 899 /905 | 897 /903 |
| Recurrence address-category sites | 78 /80 | 76 /78 |

The compiler emits two more scoreboard-wait sites inside the loop, while
removing other descriptor/control work; net recurring-body sites fall by2.
These are **static** codegen observations, not dynamic instruction counts,
saved cycles or a measured speedup. Do not turn a one-site whole-body increase
into a latency verdict. User clarified that zero added work is a preference:
correct complete-call performance decides.

Alternative screened locally: a transposed32x64 cube correctly mapped all
values but added111 static sites and2 registers. It is deprioritized relative
to microcubes, NOT proved slower on hardware. Its source/native evidence is
retained outside the production tree. Only the smaller candidate is selected
for this bounded box experiment.

## Gates and evidence

- Actual native-C producer, shipping offset helper, CuTe layout, shipping
  cube selector, actual actlize reader and native-B consumer agree on all
  2048 values. Exact-once denominator is independently fixed.
- Five host plants reject stale placement, wrong swizzle, omitted reader,
  bank collision and wrong consumer cube. Six source plants reject unpaired
  stores/loads, altered rounding and extra synchronization. Three native
  plants reject wrong transpose/copy family and a hidden lane exchange.
- Complete local run:16/16 CTests,81 host contracts,7 compiler-dialect tests,
  45 WY+61 residual algebra cases,305 original controls.27 WY images and15
  original images audited; all26 previous WY native instruction/operand
  sequences identical. Original experimental C32 spill remains separately
  labeled, not attributed to this candidate.

Local evidence: `/workspace/gdn-wy-residual-banks-evidence-20260924`;
replay `verify_local.sh`, final authority `final-local-complete.log`.
Device-library SHA256:
`b93fb50a1645a5f8d43a47ac6a705145214419786d646e9bfb6724c6b6de4e4e`.
Device numeric/BC/performance admission remains **NOT_RUN** locally.

## Box command and decision

From GDN-QSA-sm80 on `ppu-backend`, after pulling:

```bash
DEVICE=0 JOBS=16 CANDIDATE=residual-blayout \
  bash tools/run_ppu_residual_delivery_acu_box.sh
```

Set `PPU_SDK=/path/to/PPU_SDK` only if the working SDK is not the default.
No identity envs, manual paths, CSV export or inline `exit` are needed. The
runner creates its own `/workspace/...` directory, builds the candidate,
checks30 residual cases with8 repeats, requires byte equality against the
residual control and the unchanged independent2% recurrence gate, then admits
both gate values against FLA. It captures control/candidate/FLA sequentially
with `/sim/eec/shared/junfu.qx/asight/bin/acu --set full` and produces the tar
path to upload. No device profiler runs in this local handoff.

Compare all4 candidate kernels with all4 control kernels and the complete
FLA inventory including fills. Numeric failure invalidates timing. Report
state and complete sums, BC/read/write request counts, KVD-to-TSM traffic,
registers/spill and clocks. Lower BC without lower total time is NOT a win;
higher BC with faster correct full call is not automatically a loss. Keep
the result scope explicit; one capture near parity is not a stable win.
No automatic production promotion. For the independent weak-gate capture,
use the same command with `GATE=-0.1`; do not overlap captures.
