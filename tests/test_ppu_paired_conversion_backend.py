#!/usr/bin/env python3
"""All C64 extents must execute paired conversion, never an old tail fallback."""
import argparse
import os
from pathlib import Path
import torch
from test_ppu_residual_backend import admit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extension",type=Path,required=True)
    args=parser.parse_args()
    if not args.extension.is_file(): parser.error("paired-conversion extension missing")
    os.environ["GDN_QSA_WY_EXTENSION"]=str(args.extension.resolve())
    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    if "PPU" not in torch.cuda.get_device_properties(0).name.upper():
        raise RuntimeError("paired-conversion admission requires PPU")
    count=0
    for length in range(1,65):
        for gate in (-.1,-1.):
            for initial in (False,True):
                admit((1,length,1,2),gate,initial,deliveries=["full-chunk","paired-conversion"])
                count+=1
    if count!=256: raise AssertionError("paired-conversion device denominator changed")
    print("[paired conversion device] PASS cases=256 repeats=8 all64extents RAW-BIT+2%-oracle routing=UNCHANGED")


if __name__=="__main__": main()
