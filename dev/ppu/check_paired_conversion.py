#!/usr/bin/env python3
"""Bind paired NewV conversion to source, native operands and linked entry."""
import argparse
from collections import Counter
from pathlib import Path
import re
import subprocess

from check_state_pipeline import block, code
from check_full_chunk import replace_once
from host_disassembly import read_host_disassembly

ROOT = Path(__file__).resolve().parents[2]
SUBJECT = "gdn_wy_paired_conversion_state"
ENTRY = "gdn_wy_forward_residual_paired_conversion"


def expect_red(label, check, *args):
    try:
        check(*args)
    except AssertionError:
        print("[paired conversion negative]",label,"EXPECTED-RED/PASS")
    else:
        raise AssertionError("negative escaped: "+label)


def check_source(full, generic, subject, helper, binding):
    full, generic, subject, helper, binding = map(code, (full, generic, subject, helper, binding))
    loop = "for(int r=0;r<StateTile::ValueFragments;++r)"
    new_loop = block(subject, loop, "convert_new_value(x,row_decay[r])")
    expected_loop = code('''for(int r=0;r<StateTile::ValueFragments;++r) {
      float x[8]; CUTE_UNROLL for(int s=0;s<8;++s) {
        auto const rc=result_coord(lane,s);
        int const row=StateTile::value_row(warp,r)+rc.row;
        x[s]=Full || row<valid ? value[r][s] : 0.0f;
      }
      auto const converted=convert_new_value(x,row_decay[r]);
      CUTE_UNROLL for(int s=0;s<8;++s) {
        sm.value[PublishedValue::producer_offset(b_store_base,r,s)]=converted.unscaled[s];
        sm.scaled[BIntermediate::producer_offset(b_store_base,r,s)]=converted.scaled[s];
      }
    }''')
    if new_loop != expected_loop:
        raise AssertionError("paired publication changed owner, tail, order or plane")
    for is_full, control, marker in (
        (True, full, "gdn_wy_residual_full_chunk_state"),
        (False, generic, "gdn_wy_residual_gate_cache_state"),
    ):
        state = block(subject, SUBJECT+"(").replace(SUBJECT+"(",marker+"(",1)
        state = replace_once(state, new_loop, block(control,loop,"float const x=value[r][s];" if is_full
                                                     else "float const x=row<valid?value[r][s]:0.0f;"))
        for before, after in (
            ("Full?Chunk:Plan::valid(ct,p.shape.sequence)", "Chunk" if is_full else "Plan::valid(ct,p.shape.sequence)"),
            ("Full||tid<unsigned(valid)?float(p.beta[(int64_t(b)*p.shape.sequence+first+tid)*p.shape.value_heads+h]):0.0f",
             "float(p.beta[(int64_t(b)*p.shape.sequence+first+tid)*p.shape.value_heads+h])" if is_full else
             "tid<unsigned(valid)?float(p.beta[(int64_t(b)*p.shape.sequence+first+tid)*p.shape.value_heads+h]):0.0f"),
            ("BF16(Full||row<valid?beta[half]*difference:0.0f)", "BF16(beta[half]*difference)" if is_full else
             "BF16(row<valid?beta[half]*difference:0.0f)"),
        ):
            state = replace_once(state,before,after)
        if state != block(control,marker+"("):
            raise AssertionError("paired conversion changed another state operation/lifetime")
    expected_helper = code('''NewValue convert_new_value(float const (&x)[8], float const (&relative)[2]) {
      cutlass::Array<float,8> unscaled,scaled;
      CUTLASS_PRAGMA_UNROLL for(int s=0;s<8;++s) {
        unscaled[s]=x[s]; scaled[s]=x[s]*relative[s/4];
      }
      using Convert=cutlass::NumericArrayConverter<cutlass::bfloat16_t,float,8,
          cutlass::FloatRoundStyle::round_to_nearest>;
      return {Convert{}(unscaled),Convert{}(scaled)};
    }''')
    if block(helper,"NewValue convert_new_value(") != expected_helper:
        raise AssertionError("paired helper changed FP32 sources/RNE/separate rounding")
    if block(helper,"struct NewValue") != code('''struct NewValue {
        cutlass::Array<cutlass::bfloat16_t,8> unscaled;
        cutlass::Array<cutlass::bfloat16_t,8> scaled;
    }'''):
        raise AssertionError("paired helper value-plane precision/member order changed")

    host = block(subject,ENTRY+"(")
    # Reconstruct the exact old full entry; both new state instantiations have
    # identical ABI/configuration, and no tail may route around the new cast.
    host = replace_once(host,"using paired_conversion::"+SUBJECT+";",
                        "using full_chunk::gdn_wy_residual_full_chunk_state;")
    host = replace_once(host,"bool const full=gate_cache::full_chunks(sequence);int rc=solve_static::configure();",
                        "int rc;rc=solve_static::configure();")
    host = replace_once(host,'''if(full)rc=int(hggcFuncSetAttribute(gdn_wy_paired_conversion_state<true>,
      hggcFuncAttributeMaxDynamicSharedMemorySize,sizeof(gate_cache::Storage)));
      else rc=int(hggcFuncSetAttribute(gdn_wy_paired_conversion_state<false>,
      hggcFuncAttributeMaxDynamicSharedMemorySize,sizeof(gate_cache::Storage)));''',
      '''rc=int(hggcFuncSetAttribute(gdn_wy_residual_full_chunk_state,
      hggcFuncAttributeMaxDynamicSharedMemorySize,sizeof(gate_cache::Storage)));''')
    host = replace_once(host,'''if(full)gdn_wy_paired_conversion_state<true><<<grid,Plan::Threads,sizeof(gate_cache::Storage),stream>>>(p,ws,final);
      else gdn_wy_paired_conversion_state<false><<<grid,Plan::Threads,sizeof(gate_cache::Storage),stream>>>(p,ws,final);''',
      "gdn_wy_residual_full_chunk_state<<<grid,Plan::Threads,sizeof(gate_cache::Storage),stream>>>(p,ws,final);")
    host = host.replace(ENTRY+"(","gdn_wy_forward_residual_full_chunk(",1)
    old_host = block(full,"gdn_wy_forward_residual_full_chunk(")
    fallback = block(old_host,"if(!gate_cache::full_chunks(sequence))")
    old_host = old_host.replace(fallback,"",1)
    if host != old_host:
        raise AssertionError("paired host changed admission/composition/workspace/launch")
    for token in ('m.def("residual_paired_conversion",&forward<true,14,7>,',
                  'FullStages==7?'+ENTRY+':', 'FullStages<=7&&(!FullStages||(Residual&&Variant==14))'):
        if binding.count(token) != 1:
            raise AssertionError("paired Python specialization missing or wrong")


def native_publication(seq, full):
    """Symbolically trace real PR accumulators through multiply/pack/unpack/store.

    Fixed byte offsets come from the independent host layout proof. Register
    numbers are discovered, not pinned to one compiler allocation. Swapping
    native pair inputs or stores must fail even when opcode totals are equal.
    """
    pairs = [i for i,op in enumerate(seq) if op.startswith("v.pcnvt.bf16x2")]
    if len(pairs) != 8:
        raise AssertionError("native paired conversion not emitted eight times")
    start = max(i for i,op in enumerate(seq[:pairs[0]]) if op.startswith("v.mma.f32.bf16"))
    lo,hi = map(int,re.search(r"vreg\[(\d+):(\d+)\]",seq[start]).groups())
    if hi-lo != 7:
        raise AssertionError("NewV C accumulator is not eight words")
    regs = {f"vreg{lo+i}":("x",i) for i in range(8)}
    def read(r):
        r=r.removesuffix(".reuse")
        return regs.setdefault(r,("factor",r))
    stores = []
    for op in seq[start+1:]:
        name,_,body = op.partition("\t")
        if name=="s.blksyn.defer": break
        parts = [v.strip().removesuffix(".reuse") for v in body.split(",")]
        if name=="v.csel.b32":
            d,a,z,pred=parts
            if z!="0x0" or read(a)[0]!="x": raise AssertionError("unexpected tail predicate")
            regs[d]=("mask",read(a),pred)
        elif name=="v.mul.f32":
            d,a,b=parts; a,b=read(a),read(b)
            if a[0]=="factor": a,b=b,a
            if a[0] not in ("x","mask") or b[0]!="factor":
                raise AssertionError("scaled NewV no longer multiplies original FP32 input")
            regs[d]=("scale",a,b)
        elif name=="v.pcnvt.bf16x2":
            d,a,b=parts
            regs[d]=("pair",read(a),read(b))
        elif name=="v.shrl.b32":
            d,a,n=parts
            if n!="0x10" or read(a)[0]!="pair": raise AssertionError("wrong high half extraction")
            regs[d]=("half",read(a)[2])
        elif name=="tsm.st.b16":
            value=read(parts[0])
            if value[0]=="pair": value=value[1]
            elif value[0]=="half": value=value[1]
            else: raise AssertionError("store did not consume paired BF16")
            match=re.fullmatch(r"\[(0x[0-9a-f]+) \+ (vreg\d+) \* 0x1\]",parts[1])
            if not match: raise AssertionError("publication address is no longer base plus immediate")
            stores.append((match[2],int(match[1],16),value))
        elif name.startswith(("s.","smem.")) or name=="v.reg.dchk":
            pass
        else:
            raise AssertionError("unproved operation in paired conversion region: "+op)
    if len(stores)!=16: raise AssertionError("paired publication word denominator")
    offsets=(0,144,256,400,16,128,272,384)
    planes={"unscaled":{},"scaled":{}}
    bases={}; factors={}; predicates={}
    for base,offset,value in stores:
        scaled=value[0]=="scale"
        if scaled: factor=value[2];value=value[1]
        if not full:
            if value[0]!="mask": raise AssertionError("generic tail lost mask")
            pred=value[2];value=value[1]
        if value[0]!="x": raise AssertionError("pair source is not original FP32 C slot")
        slot=value[1];plane="scaled" if scaled else "unscaled"
        if bases.setdefault(plane,base)!=base or offset!=offsets[slot] or slot in planes[plane]:
            raise AssertionError("native paired half reached wrong/duplicate owner address")
        if scaled and factors.setdefault(slot//4,factor)!=factor:
            raise AssertionError("row-dependent scaling changed inside four-slot row")
        if not full and predicates.setdefault(slot//4,pred)!=pred:
            raise AssertionError("tail mask changed inside four-slot row")
        planes[plane][slot]=offset
    if set(planes["unscaled"])!=set(range(8)) or set(planes["scaled"])!=set(range(8)) or len(set(bases.values()))!=2:
        raise AssertionError("missing paired plane/slot or aliased planes")
    if len(factors)!=2 or len(set(factors.values()))!=2:
        raise AssertionError("two independent row coefficients were collapsed")
    if not full and (len(predicates)!=2 or len(set(predicates.values()))!=2):
        raise AssertionError("two independent tail row predicates were collapsed")
    return dict(paired_sites=8,unscaled_words=8,scaled_words=8,tail_predicate=not full)


def check_native(isa):
    from check_wy_binary import kernel_sequences
    kernels=kernel_sequences(isa)
    rows=[]
    for full in (True,False):
        marker=SUBJECT+"ILb"+str(int(full))+"EE"
        old_marker="gdn_wy_residual_full_chunk_stateE" if full else "gdn_wy_residual_gate_cache_stateE"
        select=lambda m:[seq for name,seq in kernels.items() if m in name]
        a,b=select(old_marker),select(marker)
        if len(a)!=1 or len(b)!=1: raise AssertionError("missing/duplicate exact paired state specialization")
        old,new=a[0],b[0]
        a,b=(Counter(op.split()[0] for op in s) for s in (old,new))
        fixed=("v.mma.","vmem.","tsm.","v.fma.f32","v.mul.f32","v.add.f32","v.exp2.","s.blksyn")
        if {k:v for k,v in a.items() if k.startswith(fixed)}!={k:v for k,v in b.items() if k.startswith(fixed)}:
            raise AssertionError("paired state changed math/traffic/publication/barriers")
        if b["v.cnvt.bf16.f32.rtte"]!=a["v.cnvt.bf16.f32.rtte"]-16 or b["v.pcnvt.bf16x2"]!=8:
            raise AssertionError("wrong scalar/paired conversion denominator")
        if any("ivreg" in op or "shuffle" in op for op in new):
            raise AssertionError("pairing introduced indirect-register/cross-lane repair")
        rows.append(dict(full=full,control_sites=len(old),candidate_sites=len(new),
                         **native_publication(new,full),device="NOT_RUN"))
    return rows


def check_linked(host, binding=False):
    from check_gate_cache_solve import functions,reachable_calls
    bodies=functions(host)
    if binding:
        roots=[n for n in bodies if "::forward<true, 14u, 7u>(" in n and not n.endswith(" [clone .cold]")]
        if len(roots)!=1: raise AssertionError("paired Python binding missing/ambiguous")
        targets={n for n in reachable_calls(bodies,roots[0]) if n.startswith("gdn_wy_forward")}
        if targets!={ENTRY}: raise AssertionError("linked Python binding selects wrong C ABI")
    else:
        calls=reachable_calls(bodies,ENTRY)
        for marker in ("solve_static::configure(","solve_static::launch_inverse(","launch_hvlayout_output(",
                       SUBJECT+"<true>(",SUBJECT+"<false>("):
            if not any(marker in n for n in calls): raise AssertionError("linked paired entry lost "+marker)
        if any(n.startswith("gdn_wy_forward_residual_full_chunk") for n in calls):
            raise AssertionError("linked paired entry bypasses its candidate")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test",action="store_true")
    for name in ("host","isa","library","binding"): parser.add_argument("--"+name,type=Path)
    args=parser.parse_args()
    paths=("csrc/gdn_chunk/gdn_wy_residual_full_chunk_ppu.cu","csrc/gdn_chunk/gdn_wy_residual_gate_cache_ppu.cu",
           "csrc/gdn_chunk/gdn_wy_paired_conversion_ppu.cu","include/gdn_qsa/ppu/wy_paired_conversion.cuh",
           "csrc/gdn_chunk/gdn_wy_ops.cpp")
    texts=[(ROOT/p).read_text() for p in paths]
    check_source(*texts)
    if args.self_test:
        for label,i,old,new in (
            ("early-rounded-scale",3,"x[s] * relative[s/4]","float(cutlass::bfloat16_t(x[s])) * relative[s/4]"),
            ("wrong-row",3,"relative[s/4]","relative[(s/4)^1]"),
            ("wrong-round",3,"round_to_nearest","round_toward_zero"),
            ("wrong-result-member-order",3,"unscaled;\n  cutlass::Array<cutlass::bfloat16_t,8> scaled;",
             "scaled;\n  cutlass::Array<cutlass::bfloat16_t,8> unscaled;"),
            ("wrong-half",2,"converted.unscaled[s]","converted.unscaled[s^1]"),
            ("wrong-plane",2,"converted.scaled[s]","converted.unscaled[s]"),
            ("missing-tail-mask",2,"Full || row < valid ? value[r][s] : 0.0f","value[r][s]"),
            ("lost-retirement",2,"__syncthreads();  // RETIRE","/* no retirement */  // RETIRE"),
            ("tail-uses-full",2,"gdn_wy_paired_conversion_state<false><<<","gdn_wy_paired_conversion_state<true><<<"),
            ("wrong-inverse-alias",2,"inverse_ws.snapshots = ws.w","inverse_ws.snapshots = ws.snapshots"),
            ("wrong-binding",4,"&forward<true, 14, 7>","&forward<true, 14>"),
        ):
            plant=texts.copy()
            if plant[i].count(old)!=1: raise AssertionError("negative seam missing/ambiguous: "+label)
            plant[i]=plant[i].replace(old,new,1)
            expect_red(label,check_source,*plant)
    if args.host:
        subprocess.run([str(args.host.resolve())],check=True)
        if args.self_test:
            bad=subprocess.run([str(args.host.resolve()),"--omit-last-tail"],capture_output=True,text=True)
            if not bad.returncode or "denominator" not in bad.stderr: raise AssertionError("omitted tail escaped actual host proof")
            print("[paired negative] omitted-tail EXPECTED-RED/PASS")
    if args.isa:
        isa=args.isa.read_text()
        print("[paired native]",check_native(isa))
        if args.self_test:
            from check_wy_binary import plant_in_kernel
            for full in (True,False):
                marker=SUBJECT+"ILb"+str(int(full))+"EE"
                for op in ("v.pcnvt.bf16x2","tsm.st.b16","v.mma.f32.bf16","s.blksyn"):
                    expect_red(marker+"-missing-"+op,check_native,plant_in_kernel(isa,marker,op,"MISSING_OPCODE"))
                own=next(s for s in isa.split("Disassembly of section ") if s.startswith(".text.kernel.") and marker in s.splitlines()[0])
                match=re.search(r"v.pcnvt.bf16x2\s+(vreg\d+), (vreg\d+), (vreg\d+)",own)
                swapped=match[0].replace(", "+match[2]+", "+match[3],", "+match[3]+", "+match[2])
                plant=isa.replace(own,own.replace(match[0],swapped,1),1)
                expect_red(marker+"-pair-half-same-count",check_native,plant)
                # Mutate the publication under test, not an earlier snapshot
                # store which happens to use the same offset in this body.
                begin=own.index("v.pcnvt.bf16x2")
                if "0x90 +" not in own[begin:]: raise AssertionError("NewV store plant missing")
                changed=own[:begin]+own[begin:].replace("0x90 +","0x80 +",1)
                plant=isa.replace(own,changed,1)
                expect_red(marker+"-store-offset-same-count",check_native,plant)
    for path,binding in ((args.library,False),(args.binding,True)):
        if path:
            host=read_host_disassembly(path)
            check_linked(host,binding)
            if args.self_test: expect_red("missing-linked-entry",check_linked,host.replace(ENTRY,"MISSING_ENTRY"),binding)
    print("[paired conversion] source=PASS native="+("PASS" if args.isa else "NOT_RUN")+" device=NOT_RUN")


if __name__=="__main__": main()
