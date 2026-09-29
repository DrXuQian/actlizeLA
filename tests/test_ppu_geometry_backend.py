#!/usr/bin/env python3
"""Every C64 tail extent crosses two chunks in every matched geometry."""
import argparse
import os
from pathlib import Path
import torch
from test_ppu_residual_backend import admit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--extension",type=Path,required=True)
    args=p.parse_args()
    if not args.extension.is_file(): p.error("geometry extension missing")
    os.environ["GDN_QSA_WY_EXTENSION"]=str(args.extension.resolve())
    torch.set_num_threads(1);torch.cuda.set_device(0)
    if "PPU" not in torch.cuda.get_device_properties(0).name.upper():
        raise RuntimeError("geometry admission requires PPU")
    count=0
    for extent in range(1,65):
        for gate in (-.1,-1.):
            for initial in (False,True):
                admit((1,64+extent,1,2),gate,initial,
                      deliveries=["full-chunk","geometry-v32-w8","geometry-v32-w4","geometry-v16-w4"])
                count+=1
    if count!=256: raise AssertionError("geometry extent denominator changed")
    print("[geometry device] PASS cases=256 repeats=8 all64extents+carried-state RAW-BIT+2%-oracle routing=UNCHANGED")


if __name__=="__main__": main()
