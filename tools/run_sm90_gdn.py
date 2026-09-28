#!/usr/bin/env python3
"""One public GDN invocation per process; CPU oracle, no warmup/device oracle.

For simulator input use --backend ppu17 --source-check and the explicit
ppu17/source-check binary. That label must never become native PPU evidence.
Wrap this application in the actual simulator command supplied for the model;
this script does not invent a simulator/profiler CLI or report Python timing.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"tests")]
import torch
from test_ppu_gdn_backend import fixture,assert_pair,digest,MAX_RELATIVE_ERROR
from actlize_la.reference.gdn_chunk_ref import torch_recurrent_gated_delta_rule
from actlize_la.gdn_sm90_interface import gdn_chunk_sm90
from actlize_la import get_device_profile
from actlize_la.device import validate_profile_options


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--extension",type=Path,required=True)
    p.add_argument("--backend",choices=("cuda_sm90","ppu17"),required=True)
    p.add_argument("--source-check",action="store_true")
    p.add_argument("--configuration",choices=("control","value64","value64-local-inverse","value128-paired"),default="control")
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--batch",type=int,default=1)
    p.add_argument("--length",type=int,default=65)
    p.add_argument("--q-heads",type=int,default=1)
    p.add_argument("--v-heads",type=int,default=2)
    p.add_argument("--gate",type=float,default=-.1)
    p.add_argument("--fp32-gate",action="store_true")
    p.add_argument("--vary-gate",action="store_true",help="distinct decay by token and head, generated on CPU")
    p.add_argument("--initial",action="store_true")
    p.add_argument("--output-only",action="store_true")
    p.add_argument("--device",type=int,default=0)
    p.add_argument("--mode",choices=("device","perfmodel"),default="device",
                   help="metadata source only; perfmodel skips device-property discovery")
    p.add_argument("--sm-count",type=int,
                   help="explicit model SM count (e.g. 20); only with --mode perfmodel")
    args=p.parse_args()
    try:
        validate_profile_options(args.mode,args.backend,args.sm_count)
    except (ValueError,RuntimeError) as error:
        p.error(str(error))
    if not args.extension.is_file(): p.error("extension missing")
    manifest_path=args.extension.parent/"build.json"
    if not manifest_path.is_file(): p.error("extension must retain its build.json receipt")
    manifest=json.loads(manifest_path.read_text())
    binary_hash=hashlib.sha256(args.extension.read_bytes()).hexdigest()
    if not manifest.get("complete") or manifest.get("extension_sha256")!=binary_hash:
        p.error("incomplete build or binary hash mismatch; do not use a stale extension")
    expected_mode="source-check" if args.source_check else "native"
    if manifest.get("target")!=args.backend or manifest.get("mode")!=expected_mode:
        p.error("requested execution target does not match build receipt")
    if manifest.get("configuration")!=args.configuration:
        p.error("requested configuration does not match build receipt")
    if not (-1. <= args.gate <= 0.): p.error("initial admission range is natural-log g in [-1,0]")
    profile=get_device_profile(args.device,mode=args.mode,backend=args.backend,sm_count=args.sm_count)
    args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    cpu=fixture(args.batch,args.length,args.q_heads,args.v_heads,args.gate)
    if args.fp32_gate: cpu=(*cpu[:3],cpu[3].float(),cpu[4])
    if args.vary_gate:
        gate = args.gate * (.25 + .75 * torch.rand(cpu[3].shape, device="cpu",
            generator=torch.Generator(device="cpu").manual_seed(101)))
        cpu=(*cpu[:3],gate.to(cpu[3].dtype),cpu[4])
    state=(torch.randn(args.batch,args.v_heads,128,128,
        generator=torch.Generator(device="cpu").manual_seed(17),device="cpu")*.005) if args.initial else None
    q,k,v,g,beta=cpu
    ratio=args.v_heads//args.q_heads
    want=torch_recurrent_gated_delta_rule(q.repeat_interleave(ratio,2),k.repeat_interleave(ratio,2),
        v,g,beta,initial_state=state,output_final_state=not args.output_only)
    os.environ["GDN_QSA_SM90_EXTENSION"]=str(args.extension.resolve())
    torch.cuda.set_device(args.device)
    device=torch.device("cuda",args.device)
    inputs=tuple(t.to(device) for t in cpu)
    initial=state.to(device) if state is not None else None
    torch.cuda.synchronize()
    print(f"[SM90 subject] backend={args.backend} source_check={args.source_check} "
          f"mode={args.mode} profile_source={profile.source} sm_count={profile.sm_count} "
          "public_calls=1 warmup=0 device_reference=0 performance=NOT_MEASURED",flush=True)
    got=gdn_chunk_sm90(*inputs,initial_state=initial,output_final_state=not args.output_only,
                      backend=args.backend,source_check=args.source_check,configuration=args.configuration)
    torch.cuda.synchronize()
    actual=tuple(x.detach().cpu() for x in got if x is not None)
    expected=tuple(x for x in want if x is not None)
    errors=assert_pair(actual,expected)
    if args.output_only and got[1] is not None: raise AssertionError("unexpected final state")
    # Oracle negatives consume only the captured CPU output, never rerun GPU.
    for index in range(len(actual)):
        bad=list(actual); bad[index]=torch.zeros_like(bad[index],device="cpu")
        try: assert_pair(bad,expected)
        except AssertionError: pass
        else: raise AssertionError("zero-output/state negative escaped")
    if digest(tuple(t.cpu() for t in inputs))!=digest(cpu): raise AssertionError("input mutation")
    if state is not None and not torch.equal(initial.cpu(),state): raise AssertionError("initial state mutation")
    result=dict(backend=args.backend,configuration=args.configuration,source_check=args.source_check,device=profile.name,
        mode=args.mode,device_profile=asdict(profile),
        shape=[args.batch,args.length,args.q_heads,args.v_heads,128],gate=args.gate,
        gate_pattern="token-head-distinct" if args.vary_gate else "constant",
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        errors=errors,limit=MAX_RELATIVE_ERROR,input_sha256=digest(cpu),output_sha256=digest(actual),
        extension_sha256=binary_hash,source_sha256=manifest['source_sha256'],
        public_calls=1,trace_kernel_count="REQUIRES_TRACE_VERIFICATION",verdict="NUMERIC/PASS",
        performance="NOT_MEASURED",replay="NOT_RUN_SINGLE_CALL")
    (args.out/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    print("[SM90 result] "+json.dumps(result),flush=True)


if __name__=="__main__": main()
