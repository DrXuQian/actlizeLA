#!/usr/bin/env python3
"""Matched state geometry: source math, native inventory and linked dispatch."""
import argparse
from collections import Counter
from pathlib import Path
import re

from check_state_pipeline import block, code
from check_full_chunk import replace_once

ROOT=Path(__file__).resolve().parents[2]
GEOMETRIES=((32,8),(32,4),(16,4))
SUBJECT="gdn_wy_geometry_state"


def check_source(subject, full, generic, binding):
    subject,full,generic,binding=map(code,(subject,full,generic,binding))
    for complete,old,name in ((True,full,"gdn_wy_residual_full_chunk_state"),
                              (False,generic,"gdn_wy_residual_gate_cache_state")):
        state=block(subject,SUBJECT+"(")
        aliases=state[state.index("usingGeometry="):state.index("extern__shared__")]
        if aliases!=code('''using Geometry=Config<Columns,Warps>;
          using StateTile=typename Geometry::StateTile; using Plan=typename Geometry::Plan;
          using Key=typename Geometry::Key; using Inverse=typename Geometry::Inverse;
          using Value=typename Geometry::Value; using Snapshot=typename Geometry::Snapshot;
          using BIntermediate=typename Geometry::BIntermediate;
          using PublishedValue=typename Geometry::PublishedValue; constexpr int ValueTile=Columns;'''):
            raise AssertionError("geometry body uses unmatched owner/layout aliases")
        state=state.replace(aliases, "", 1).replace(SUBJECT+"(",name+"(",1)
        state=state.replace("typenameGeometry::Storage", "gate_cache::Storage")
        state=state.replace("::templatepublish<", "::publish<").replace("Key::templateload<", "Key::load<")
        for before,after in (
          ("Full?Chunk:Plan::valid(ct,p.shape.sequence)","Chunk" if complete else "Plan::valid(ct,p.shape.sequence)"),
          ("Full||tid<unsigned(valid)?float(p.beta[(int64_t(b)*p.shape.sequence+first+tid)*p.shape.value_heads+h]):0.0f",
           "float(p.beta[(int64_t(b)*p.shape.sequence+first+tid)*p.shape.value_heads+h])" if complete else
           "tid<unsigned(valid)?float(p.beta[(int64_t(b)*p.shape.sequence+first+tid)*p.shape.value_heads+h]):0.0f"),
          ("BF16(Full||row<valid?beta[half]*difference:0.0f)","BF16(beta[half]*difference)" if complete else
           "BF16(row<valid?beta[half]*difference:0.0f)"),
          ("Full||row<valid?value[r][s]:0.0f", "value[r][s]" if complete else "row<valid?value[r][s]:0.0f")):
            state=replace_once(state,before,after)
        if state!=block(old,name+"("):
            raise AssertionError("geometry changed nongeometry state math/lifetime: full="+str(complete))
    host=block(subject,"int launch(")
    for token in ("inverse_ws.snapshots=ws.w;", "solve_static::launch_inverse(p,inverse_ws,stream)",
                  "bool const full=gate_cache::full_chunks(sequence);",
                  "unsigned const grid=unsigned(int64_t(batch)*value_heads*(Dim/Columns));",
                  "residual_warps8_hvlayout::launch_hvlayout_output(p,ws,static_cast<BF16*>(output),stream)"):
        if host.count(code(token))!=1: raise AssertionError("geometry composition/launch changed")
    for full_flag in ("true","false"):
        if host.count(f"gdn_wy_geometry_state<Columns,Warps,{full_flag}>")!=2:
            raise AssertionError("missing geometry full/generic configure+launch")
        prefix="if(full)" if full_flag=="true" else "else"
        if code(prefix+f"gdn_wy_geometry_state<Columns,Warps,{full_flag}><<<grid,Warps*32,sizeof(Storage),stream>>>(p,ws,final);") not in host:
            raise AssertionError("geometry tail was dispatched to the full body")
    for i,(v,w) in enumerate(GEOMETRIES,1):
        entry=f"gdn_wy_forward_geometry_v{v}_w{w}"
        export=block(subject,entry+"(")
        if f"returngdn_qsa::wy::geometry::launch<{v},{w}>(" not in export:
            raise AssertionError("C entry selected another geometry")
        if f'"residual_geometry_v{v}_w{w}",&forward<true,14,0,{i}>' not in binding:
            raise AssertionError("Python entry selected another geometry")
        if f"Geometry=={i}?{entry}:" not in binding:
            raise AssertionError("binding selected another launcher")
    if "Variant==3||Geometry==3?8:4" not in binding:
        raise AssertionError("V16 grid overflow bound lost")


def check_native(isa):
    from check_wy_binary import kernel_sequences
    seqs=kernel_sequences(isa)
    rows=[]
    for v,w in GEOMETRIES:
        for full in (True,False):
            marker=f"{SUBJECT}ILi{v}ELi{w}ELb{int(full)}EE"
            own=[(n,s) for n,s in seqs.items() if marker in n]
            if len(own)!=1: raise AssertionError("geometry image denominator: "+marker)
            name,seq=own[0]; counts=Counter(op.split()[0] for op in seq)
            vf=64*(v//16)//(16*w); kf=2*vf
            totals=(sum(n for op,n in counts.items() if op.startswith("v.mma.f32.bf16")),
                    sum(n for op,n in counts.items() if op.startswith("tsm.ld.swzl")),
                    counts["vmem.aiu.ld.tsm.l0.t0.p0.s0.m0.2d.b16.kp1"])
            if totals != (12*vf+4*kf,16+12*vf+4*kf,4):
                raise AssertionError(f"geometry native MMA/ld.swzl/AIU mismatch {v}/{w}: {totals}")
            if any(op.startswith(("v.shuffle", "v.mma.f32.tf32", "local.")) for op in counts):
                raise AssertionError("geometry introduced reduction shuffle/emulated arithmetic")
            if any(op.startswith(("vmem.ld.tsm","vmem.aiu.ld.tsm.l1","tsm.ld.ncom")) for op in counts):
                raise AssertionError("geometry lost matched AIU.swzl + ld.swzl")
            if any("ivreg" in op for op in seq) or "vmem.st.b32x4" not in counts or "vmem.st.b16" in counts:
                raise AssertionError("geometry lost direct fragments/vector publication")
            if counts["s.blksyn.defer"]!=5 or sum("commit_group(0)" in op for op in seq)!=1:
                raise AssertionError("geometry native lifetime barriers changed")
            oldmarker="gdn_wy_residual_full_chunk_stateE" if full else "gdn_wy_residual_gate_cache_stateE"
            old=[s for n,s in seqs.items() if oldmarker in n]
            if len(old)!=1: raise AssertionError("geometry control missing")
            # Function ordinal in disassembler BB labels changes between TUs;
            # preserve block number, actual branch displacement and operands.
            normalized=lambda s:[re.sub(r"<BB\d+_(\d+)>",r"<BB_\1>",op) for op in s]
            rows.append(dict(V=v,warps=w,full=full,symbol=name,static_instructions=len(seq),
                             control_identical=(seq==old[0]) if (v,w)==(32,8) else None,
                             control_native_identical=(normalized(seq)==normalized(old[0])) if (v,w)==(32,8) else None))
    return rows


def check_linked(host,binding=False):
    from check_gate_cache_solve import functions,reachable_calls
    bodies=functions(host)
    for i,(v,w) in enumerate(GEOMETRIES,1):
        entry=f"gdn_wy_forward_geometry_v{v}_w{w}"
        if binding:
            roots=[n for n in bodies if f"::forward<true, 14u, 0u, {i}u>(" in n and not n.endswith(" [clone .cold]")]
            if len(roots)!=1: raise AssertionError("geometry Python entry missing/ambiguous")
            targets={n for n in reachable_calls(bodies,roots[0]) if n.startswith("gdn_wy_forward")}
            if targets!={entry}: raise AssertionError("geometry Python entry calls wrong C ABI")
        else:
            calls=reachable_calls(bodies,entry)
            helpers=[n for n in calls if n.startswith(f"int gdn_qsa::wy::geometry::launch<{v}, {w}>(")]
            if len(helpers)!=1: raise AssertionError("geometry entry lost its exact launch specialization")
            # HGGC exposes this template through a weak PLT. Follow only the
            # selected instantiation, not unrelated library stage internals.
            calls |= reachable_calls(bodies,helpers[0])
            for marker in ("solve_static::configure(","solve_static::launch_inverse(","launch_hvlayout_output(",
                           f"{SUBJECT}<{v}, {w}, true>(",f"{SUBJECT}<{v}, {w}, false>("):
                if not any(marker in n for n in calls): raise AssertionError("geometry linked entry lost "+marker)
            if any("gdn_wy_forward_residual" in n for n in calls):
                raise AssertionError("geometry silently bypassed candidate")


def main():
    p=argparse.ArgumentParser();p.add_argument("--isa",type=Path);p.add_argument("--self-test",action="store_true")
    p.add_argument("--library",type=Path);p.add_argument("--binding",type=Path)
    a=p.parse_args()
    files=[ROOT/"csrc/gdn_chunk"/n for n in ("gdn_wy_geometry_ppu.cu","gdn_wy_residual_full_chunk_ppu.cu",
                                                "gdn_wy_residual_gate_cache_ppu.cu","gdn_wy_ops.cpp")]
    sources=[p.read_text() for p in files];check_source(*sources)
    if a.self_test:
        for before,after in (("BF16(x * row_decay", "BF16(float(BF16(x)) * row_decay"),
                             ("StateTile::k_row(warp, k)","StateTile::k_row(warp, 0)"),
                             ("Dim / Columns","Dim / 32"),
                             ("if (full) gdn_wy_geometry_state", "if (true) gdn_wy_geometry_state")):
            changed=sources.copy();changed[0]=changed[0].replace(before,after,1)
            if changed==sources: raise AssertionError("plant target absent")
            try: check_source(*changed)
            except AssertionError: print("[geometry source negative]",before,"EXPECTED-RED/PASS")
            else: raise AssertionError("geometry source negative escaped")
    if a.isa:
        isa=a.isa.read_text()
        for row in check_native(isa): print("[geometry native]",row)
        if a.self_test:
            from check_wy_binary import plant_in_kernel
            for v,w in GEOMETRIES:
                for full in (0,1):
                    marker=f"{SUBJECT}ILi{v}ELi{w}ELb{full}EE"
                    planted=plant_in_kernel(isa,marker,"v.mma.f32.bf16","MISSING_MMA")
                    try: check_native(planted)
                    except AssertionError: print("[geometry native negative]",marker,"EXPECTED-RED/PASS")
                    else: raise AssertionError("geometry native negative escaped")
    for path,binding in ((a.library,False),(a.binding,True)):
        if path:
            from host_disassembly import read_host_disassembly
            host=read_host_disassembly(path);check_linked(host,binding)
            if a.self_test:
                for v,w in GEOMETRIES:
                    try: check_linked(host.replace(f"gdn_wy_forward_geometry_v{v}_w{w}","MISSING_GEOMETRY"),binding)
                    except AssertionError: print("[geometry linked negative]",v,w,"EXPECTED-RED/PASS")
                    else: raise AssertionError("geometry linked negative escaped")
    print("[geometry source/native] PASS device=NOT_RUN")


if __name__=="__main__": main()
