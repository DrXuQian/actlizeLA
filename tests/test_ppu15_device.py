#!/usr/bin/env python3
"""Public-entry admission on ZW890; independent recurrence, original fixed 2%."""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from actlize_la import gdn_forward
from actlize_la.serving.device import validate_device_target
from test_ppu_gdn_backend import fixture, assert_pair, digest, print_failure, MAX_RELATIVE_ERROR
from test_ppu_wy_backend import oracle


def run_case(shape, gate, nonzero, variable=False):
    cpu = list(fixture(*shape,gate))
    if nonzero:
        cpu[3] = cpu[3].float()
    if variable:
        cpu[1] = torch.nn.functional.normalize(cpu[1].float(),dim=-1).bfloat16()
        cpu[3] = torch.linspace(-.06,-.001,shape[1])[None,:,None].expand(shape[0],shape[1],shape[3]).contiguous()
        cpu[4] = torch.linspace(.05,.95,shape[1])[None,:,None].expand(shape[0],shape[1],shape[3]).bfloat16().contiguous()
    initial = torch.randn(shape[0],shape[3],128,128,generator=torch.Generator().manual_seed(18))*.005 if nonzero else None
    expected = oracle(cpu,initial)
    inputs = tuple(x.cuda() for x in cpu)
    state = initial.cuda() if initial is not None else None
    def call(inputs=inputs,final=True):
        return gdn_forward(*inputs,initial_state=state,output_final_state=final,backend="ppu15")
    got = call()
    torch.cuda.synchronize()
    try:
        errors = assert_pair(got,expected)
    except AssertionError:
        print_failure(f"ppu15/{shape}/{gate}/{nonzero}",got,expected)
        raise
    if got[0].dtype != torch.bfloat16 or got[1].dtype != torch.float32:
        raise AssertionError("output/state precision contract changed")
    fingerprint = digest(got)
    for _ in range(7):
        if digest(call()) != fingerprint:
            raise AssertionError("output/state fingerprints changed across 8 launches")
    ratio=shape[3]//shape[2]
    expanded=(inputs[0].repeat_interleave(ratio,2),inputs[1].repeat_interleave(ratio,2),*inputs[2:])
    if digest(call(expanded)) != fingerprint:
        raise AssertionError("GVA does not equal explicit head expansion")
    out, absent = call(final=False)
    if absent is not None or digest((out,)) != digest((got[0],)):
        raise AssertionError("output-only contract changed")
    if digest(inputs) != digest(cpu) or (state is not None and not torch.equal(state.cpu(),initial)):
        raise AssertionError("forward modified inputs")
    for index in (0,1):
        wrong = list(got)
        wrong[index] = torch.zeros_like(wrong[index])
        try:
            assert_pair(wrong,expected)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"zero-{index} negative escaped original 2% gate")
    record=dict(shape=shape,gate=gate,initial=nonzero,variable=variable,errors=errors,
                repeats=8,fingerprint=fingerprint,status="PASS")
    print("[ppu15 forward] "+json.dumps(record),flush=True)
    return record


def measure(samples, launches, warmup):
    records=[]
    for shape in ((1,2048,16,32),(1,2048,16,64),(2,2048,16,32)):
        for gate in (-.1,-1.):
            cpu=fixture(*shape,gate)
            expected=oracle(cpu,None)
            inputs=tuple(t.cuda() for t in cpu)
            call=lambda: gdn_forward(*inputs,backend="ppu15")
            reference=call()
            errors=assert_pair(reference,expected)
            fingerprint=digest(reference)
            for _ in range(warmup):
                call()
            torch.cuda.synchronize()
            raw=[]
            for _ in range(samples):
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(launches):
                    result=call()
                end.record(); end.synchronize()
                raw.append(start.elapsed_time(end)*1000/launches)
            if digest(result)!=fingerprint:
                raise AssertionError("post-timing fingerprint differs")
            record=dict(shape=shape,gate=gate,median_us=statistics.median(raw),samples_us=raw,
                launches=launches,protocol="full-forward-device-event-span-upper-includes-host-idle",
                fingerprint=fingerprint,device="ZW890",errors=errors,
                numerical_scope="same-fixture-independent-recurrence-plus-post-timing-replay")
            print("[ppu15 perf] "+json.dumps(record),flush=True)
            records.append(record)
    return records


def correctness_cases():
    cases = [((2, length, 1, 2), gate, initial, False)
             for length in (1, 63, 64, 65, 127, 128, 129)
             for gate in (-.1, -1.) for initial in (False, True)]
    cases += [((1, 129, 2, 2), -.1, True, False),
              ((1, 129, 2, 4), -.01, True, True)]
    cases += [((1, length, 16, 32), gate, False, False)
              for length in (2048, 2049) for gate in (0., -.001, -.1, -1.)]
    cases += [((1, 2048, 16, 32), -.1, True, False),
              ((1, 2048, 16, 64), -.1, False, False)]
    # Include GVA1:3 and long contexts; never substitute an810 numerical result.
    for length in (2048, 8192):
        for hv in (32, 48):
            for gate in (-.1, -1.):
                cell = ((1, length, 16, hv), gate, True, False)
                if cell not in cases:
                    cases.append(cell)
    if len(cases) != 47 or len(set(cases)) != 47:
        raise AssertionError("coverage denominator must be47 unique cases")
    return cases


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--extension",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--samples",type=int,default=7)
    p.add_argument("--launches",type=int,default=20)
    p.add_argument("--warmup",type=int,default=5)
    p.add_argument("--no-perf",action="store_true")
    args=p.parse_args()
    if not args.extension.is_file() or min(args.samples,args.launches)<1 or args.warmup<0:
        p.error("missing extension or invalid sample counts")
    os.environ["GDN_QSA_PPU15_EXTENSION"]=str(args.extension.resolve())
    torch.set_num_threads(1); torch.cuda.set_device(0)
    props=torch.cuda.get_device_properties(0)
    validate_device_target("ppu15", props.name)
    identity=dict(name=props.name,cu=props.multi_processor_count,torch=torch.__version__,runtime=torch.version.cuda,
                  extension=str(args.extension.resolve()),criterion=MAX_RELATIVE_ERROR)
    print("[ppu15 identity] "+json.dumps(identity),flush=True)
    cases=[run_case(*cell) for cell in correctness_cases()]
    if len(cases)!=47:
        raise AssertionError("coverage denominator changed: expected47")
    args.out.mkdir(parents=True,exist_ok=True)
    result=dict(identity=identity,cases=cases,device_admission="PASS",performance=[],
                performance_status="NOT_RUN" if args.no_perf else "PENDING")
    (args.out/"device.json").write_text(json.dumps(result,indent=2)+"\n")
    if not args.no_perf:
        result["performance"]=measure(args.samples,args.launches,args.warmup)
        result["performance_status"]="MEASURED_NOT_PERFORMANCE_ADMITTED"
    (args.out/"device.json").write_text(json.dumps(result,indent=2)+"\n")
    print("[ppu15 device] PASS cases=47 repeats=8 GVA/output-only/input-immutable; original-2%-gate=UNCHANGED; performance="+result["performance_status"])


if __name__ == "__main__":
    main()
