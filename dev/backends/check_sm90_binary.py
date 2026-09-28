#!/usr/bin/env python3
"""Check every assembled CUDA SM90a body; never call it PPU native evidence."""
import argparse
import json
from pathlib import Path
import re
import subprocess


def inspect(sass):
    if not re.search(r"arch = sm_90a",sass): raise ValueError("not an assembled SM90a image")
    pieces=re.split(r"Function\s*:\s*(\S+)",sass)
    bodies=dict(zip(pieces[1::2],pieces[2::2]))
    kernels={name:body for name,body in bodies.items() if "FlatKernelTmaWarpSpecializedKdaFwd" in name}
    if len(kernels)!=4: raise ValueError(f"need all four gate/initial-state specializations, got {len(kernels)}")
    result={}
    for name,body in kernels.items():
        counts={op:len(re.findall(r"\b"+op+r"(?:\.|\s)",body))
                for op in ("HGMMA","HMMA","UTMALDG","UTMASTG","STG","LDL","STL")}
        for op in ("HGMMA","HMMA","UTMALDG","UTMASTG","STG"):
            if not counts[op]: raise ValueError(f"{name}: missing live {op}")
        # A remaining BF16 tail store must not hide a deleted FP32 state path,
        # or vice versa. These are the admitted compiler's actual op widths.
        counts["state_stores"]=len(re.findall(r"\bSTG\.E(?:\.64|\.128)?\s",body))
        counts["tail_stores"]=len(re.findall(r"\bSTG\.E\.U16\s",body))
        if not counts["state_stores"] or not counts["tail_stores"]:
            raise ValueError(f"{name}: missing FP32 state or BF16 tail store")
        result[name]=counts
    return result


def negatives(sass):
    first=re.search(r"Function\s*:\s*(\S+)",sass).group(1)
    plants=[sass.replace(first,"omitted-entry",1)]
    for op in ("HGMMA","UTMALDG","UTMASTG","STG"):
        pieces=re.split(r"(Function\s*:\s*\S+)",sass)
        pieces[2]=re.sub(r"\b"+op+r"(?=\.|\s)","REMOVED",pieces[2])
        plants.append("".join(pieces))
    for pattern in (r"\bSTG\.E(?:\.64|\.128)?\s",r"\bSTG\.E\.U16\s"):
        pieces=re.split(r"(Function\s*:\s*\S+)",sass)
        pieces[2]=re.sub(pattern,"REMOVED ",pieces[2]);plants.append("".join(pieces))
    for plant in plants:
        try: inspect(plant)
        except ValueError: pass
        else: raise AssertionError("a missing kernel/instruction path escaped")
    return len(plants)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("binary",type=Path);p.add_argument("--cuobjdump",default="/usr/local/cuda/bin/cuobjdump")
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    sass=subprocess.check_output([a.cuobjdump,"--dump-sass",str(a.binary)],text=True)
    resources=subprocess.check_output([a.cuobjdump,"--dump-resource-usage",str(a.binary)],text=True)
    (a.out/"image.sass").write_text(sass);(a.out/"resources.txt").write_text(resources)
    result=dict(kernels=inspect(sass),negative_controls=negatives(sass),scope="CUDA-SM90-ASSEMBLY-NOT-PPU-EXECUTION")
    (a.out/"codegen.json").write_text(json.dumps(result,indent=2)+"\n")
    print(f"[SM90 codegen] PASS kernels={len(result['kernels'])} negatives={result['negative_controls']} "
          "live=WGMMA+TF32+TMA-load+TMA-store+state-store; device=NOT_RUN")


if __name__=="__main__":main()
