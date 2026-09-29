#!/usr/bin/env python3
"""Bounded matched-geometry captures using the existing ACU receipt contract."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile

from collect_ppu_gdn_acu import ROOT, copy_file, read_wy_run, sha

DELIVERIES=("residual-geometry-v32-w8","residual-geometry-v32-w4","residual-geometry-v16-w4")
CONTROL="residual-full-chunk"
GATES=(-1.0,-0.1)


def load_shapes(path):
    shapes=json.loads(path.read_text())
    if not isinstance(shapes,list) or len(shapes)!=6:
        raise ValueError("geometry denominator must be six registered shapes")
    for s in shapes:
        if not isinstance(s,dict) or set(s)!={"B","S","Hk","Hv"} or any(type(v)is not int or v<=0 for v in s.values()) or s["Hv"]%s["Hk"]:
            raise ValueError("invalid explicit shape/GVA")
    if len({tuple(s[k] for k in ("B","S","Hk","Hv")) for s in shapes})!=6:
        raise ValueError("duplicate geometry shape")
    return shapes


def cells(shapes):
    for s in shapes:
        label="B{B}-S{S}-Hk{Hk}-Hv{Hv}".format(**s)
        for gate in GATES:
            yield f"acu-{label}-g{gate}",s,gate


def command(run,shape,gate):
    return [sys.executable,str(ROOT/"tools/collect_ppu_gdn_acu.py"),"--wy-run",str(run),
            "--wy-control",CONTROL,"--wy-delivery",DELIVERIES[0],"--wy-extra-deliveries",*DELIVERIES[1:],
            "--batch",str(shape["B"]),"--sequence",str(shape["S"]),"--q-heads",str(shape["Hk"]),
            "--value-heads",str(shape["Hv"]),"--gate",str(gate)]


def check_comparison(comparison,shapes):
    expected={(s["B"],s["S"],s["Hk"],s["Hv"],g) for _,s,g in cells(shapes)}
    actual=[]
    for c in comparison["cases"]:
        s=c["shape"]
        actual.append((s["B"],s["S"],s["Hk"],s["Hv"],c["g"]))
        if not {"wy-"+d for d in (*DELIVERIES,CONTROL)}<=set(c["arms"]):
            raise ValueError("geometry comparison did not admit every candidate/control")
    if len(actual)!=12 or set(actual)!=expected:
        raise ValueError("geometry comparison shape/gate denominator mismatch")


def pack(run,shapes):
    files=["sha.txt","source.diff","submodules.txt","binaries.sha256","comparison.json","comparison.log",
           "residual-correctness.log","geometry-edge.log","geometry-native.log","l040_wy_geometry.log",
           "geometry-shapes.json","geometry-registration.md","codegen.log"]
    source_sha=(run/"sha.txt").read_text().strip()
    if not re.fullmatch(r"[0-9a-f]{40}",source_sha): raise ValueError("missing measured source SHA")
    identity=None
    for name,shape,gate in cells(shapes):
        bundle=run/name/"bundle"
        status=json.loads((bundle/"STATUS.json").read_text())
        expected_order=["wy-control","wy",*["wy-"+d for d in DELIVERIES[1:]],"fla"]
        if status.get("status")!="PASS" or status.get("errors") or status.get("capture_order")!=expected_order or set(status.get("capture_arms",{}))!=set(expected_order):
            raise ValueError("incomplete/missing geometry capture: "+name)
        if status.get("comparison_origin",{}).get("source_sha")!=source_sha:
            raise ValueError("mixed source SHA: "+name)
        for label,delivery in zip(expected_order,(CONTROL,*DELIVERIES,DELIVERIES[0])):
            arm=status["capture_arms"][label]
            expected=dict(role="fla" if label=="fla" else "wy",wy_delivery=delivery,
                          directory="." if label in ("wy","fla") else label)
            if arm!=expected: raise ValueError("wrong/missing arm mapping: "+name+"/"+label)
            receipt=json.loads((bundle/arm["directory"]/(arm["role"]+".json")).read_text())
            if receipt["shape"]!=dict(**shape,K=128,V=128) or receipt["gate"]!=gate:
                raise ValueError("wrong captured shape/gate: "+name)
            current=(receipt["device"],receipt["torch"],receipt["extension_sha256"],receipt["library_sha256"])
            if identity is None: identity=current
            elif current!=identity: raise ValueError("mixed device/runtime/binary across geometry cells")
        archive=run/name/(name+".tar.gz")
        record=archive.with_suffix(archive.suffix+".sha256").read_text().split()
        if len(record)!=2 or record[0]!=sha(archive) or record[1]!=archive.name:
            raise ValueError("capture archive hash mismatch: "+name)
        # Bind the status/receipts inspected here to the actual upload bytes,
        # not a post-pack edit of the working capture directory.
        with tarfile.open(archive,"r:gz") as tar:
            authority=["STATUS.json",*[
                str(Path(a["directory"])/(a["role"]+".json")) for a in status["capture_arms"].values()]]
            for relative in authority:
                key=name+"/"+relative
                members=[m for m in tar.getmembers() if m.name==key]
                if len(members)!=1 or not members[0].isfile():
                    raise ValueError("missing/duplicate archived receipt: "+key)
                archived=hashlib.sha256(tar.extractfile(members[0]).read()).hexdigest()
                if archived!=sha(bundle/relative): raise ValueError("archived receipt differs: "+key)
        files.append(str(archive.relative_to(run)))
    for name in files:
        p=run/name
        if not p.is_file() or p.is_symlink() or (p.stat().st_size==0 and name!="source.diff"):
            raise ValueError("missing/empty/nonregular evidence: "+name)
        if any(parent.is_symlink() for parent in p.parents if parent!=run.parent):
            raise ValueError("symlink in evidence path: "+name)
    archive=run/"geometry.tar.gz"; manifest=run/"geometry.SHA256SUMS"
    if archive.exists() or archive.is_symlink() or manifest.exists() or manifest.is_symlink():
        raise ValueError("preserve existing geometry archive/manifest")
    with manifest.open("x") as f:
        f.write("".join(f"{sha(run/name)}  {name}\n" for name in files))
    with archive.open("xb") as raw,tarfile.open(fileobj=raw,mode="w:gz") as tar:
        for name in [manifest.name,*files]: tar.add(run/name,arcname=name,recursive=False)
    print(f"[geometry] captures=12 arms=60 source_sha={source_sha} UPLOAD={archive} sha256={sha(archive)}",flush=True)
    return archive


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("run",type=Path)
    p.add_argument("--pack-only",action="store_true")
    args=p.parse_args();run=args.run.resolve()
    authority=run/"geometry-shapes.json"
    if not args.pack_only:
        if authority.exists(): raise ValueError("geometry sweep already started; artifacts preserved")
        copy_file(ROOT/"dev/ppu/geometry_shapes.json",authority)
        copy_file(ROOT/"docs/PPU10_GEOMETRY.md",run/"geometry-registration.md")
    shapes=load_shapes(authority)
    _,comparison,_=read_wy_run(run);check_comparison(comparison,shapes)
    if not args.pack_only:
        for i,(name,shape,gate) in enumerate(cells(shapes),1):
            env=os.environ.copy();env.pop("EXTENSION",None);env["OUT"]=str(run/name)
            print(f"[geometry] cell={i}/12 shape={shape} gate={gate}",flush=True)
            subprocess.run(command(run,shape,gate),env=env,cwd=ROOT,check=True)
    pack(run,shapes)
    print("[geometry] numerics/capture complete; default selection awaits ACU analysis/repeats",flush=True)


if __name__=="__main__": main()
