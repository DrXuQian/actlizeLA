# S36:32-cell stage sweep, no new winner

Keep S24 (`bc3c154`, measured binary `43d62a8...`). The goal of beating
fastest FlashInfer no-CP remains **NOT MET**. H800 stays on and the next
bounded experiment changes V-column ownership, not more stage depths.

Fixed B1/T2048/Hqk16/Hv32/K128/V128/C64; gates -0.1/-1; BF16 operands,
FP32 final state. Same idle H800 PCIe/114 SMs, CUDA12.8.93. No tolerance,
fastmath, input ABI, shipping selector, or actual SM80 backend changes.

## Complete inventory, not representative sampling

Q=2 × K={2,3} × V={1,2} × O={1,2} × alpha={2,5} × beta={2,5}:
**32/32 compiled,32/32 native PASS,32/32 numerical PASS,32/32 screened,
0 rejected,0 pending**. Actual four-specialization shared storage is
168960..224256B, below the queried232448B capacity. The default cell0
matches all27,856 S24 native instructions **and operands** exactly.
Every other cell preserves matrix and completion families; ring-init counts
are tied to the actual compiled depths. No C7512/C7510 serialization.

Each cell passes the same14 CPU O/state cases (<2%), with identical parent
input/output/error fingerprints, two direct-byte overflow stresses, and
repeated raw-equal outputs. This is448 numerical cases plus64 stress pairs.
The actual type probe, missing-cell/duplicate/wrong-stage negatives and
native missing-body/relaxed-wait negatives are retained. Host suite41 PASS.

Graph events only selected the registered top2 per gate; union={4,16}.
They were almost level with control in events, but **did not win nsys**.
Do not promote those small event differences or call this a global config
optimum: geometry, state WG count and register-role budgets were fixed.

## Finalists: complete-forward nsys kernel sums

Twelve interleaved/reversed calls per role; unchanged disjoint-range rule.
All288 captured forwards are re-extracted exactly from their four SQLite
traces. Candidate/raw checks and DeviceWatch are clean in every capture.

| Change from S24 | gate | candidate median [range],us | paired S24,us | fastest FI no-CP,us | versus S24 |
|---|---:|---:|---:|---:|---|
| O stages1→2 | -0.1 |124.002 [122.146,124.386]|120.5615|112.786|CONTROL-WINS|
| O stages1→2 | -1 |122.5615 [120.354,123.298]|119.538|112.2415|UNRESOLVED|
| beta stages2→5 | -0.1 |121.8415 [121.057,122.274]|120.0815|112.4015|UNRESOLVED|
| beta stages2→5 | -1 |122.7535 [121.634,123.938]|120.674|112.882|CONTROL-WINS|

Fastest FI wins all four comparisons with disjoint ranges. No new QLA
capture: these candidates did not pass the more demanding FI target.
Prior S24 QLA wins remain historical, not remeasured here. Two finalists
compile against PPU CUTLASS3.6 in **CUDA source-check** mode; nativePPU1.7
SDK/model unavailable is **SKIP**, not device PASS.

## Bound evidence and next hypothesis

Source branch `sm90-stage-sweep-20260926`, kernel/plumbing `4d84cfd`,
measurement continuation `8f513e2`. The branch's
`tools/record_sm90_stage_sweep.py` rechecks all32 binaries/native types,
numerical cases/stress bytes and all4 raw SQLite reports without changing
the analyzer's thresholds. Full cells/samples/hashes:
`/workspace/gdn-sm90-stage-sweep-20260926/campaign.json`.

Local and remote raw archive:
`/workspace/gdn-sm90-stage-sweep-20260926/evidence-stage-20260926T1503Z.tar.gz`
SHA256 `d15838fc845eb5a9dd7be9a1eab891ce86f20862404761247f70d65e98fb5c7d`.
Includes every build, admission, screen and all final profiler records.

S37/S38 are separately registered before edits: split V128 into two disjoint
V64 CTAs, each with one state WG, then independently raise the freed auxiliary
register budget104→232. State recurrence and arithmetic stay unchanged; no
partial reduction. QK/inverse work and Q/K traffic **are duplicated**. This
tests a resource/parallelism tradeoff, not a claim that more CTAs are free.
