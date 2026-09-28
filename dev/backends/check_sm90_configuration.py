#!/usr/bin/env python3
"""Preserve the four actual native bodies/resources of one retained build."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
from verify_sm90_native_pair import kernels

ROOT=Path(__file__).resolve().parents[2]
ANCHORS=Path(__file__).with_name("sm90_configuration_anchors.json")


def verify(native, resources, authority):
    if len(native)!=4 or resources!=authority["resources"]:
        raise AssertionError("configuration resource/body denominator changed")
    digest=hashlib.sha256(json.dumps(native,sort_keys=True).encode()).hexdigest()
    if digest!=authority["native_sha256"]:
        raise AssertionError("configuration native instruction/order/operand stream changed")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("build",type=Path)
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args()
    receipt=json.loads((a.build/"build.json").read_text())
    if not receipt.get("complete") or receipt.get("target")!="cuda_sm90":
        raise AssertionError("requires completed native CUDA build, not PPU source-check")
    authority=json.loads(ANCHORS.read_text())["configurations"][receipt["configuration"]]
    obj=a.build/"launch.o"
    if hashlib.sha256(obj.read_bytes()).hexdigest()!=receipt["device_object_sha256"]:
        raise AssertionError("object no longer matches receipt")
    dumper=Path(receipt["compiler"]).parent/"cuobjdump"
    for flag,filename in (("--dump-sass","image.sass"),("--dump-resource-usage","resources.txt")):
        actual=subprocess.check_output([str(dumper),flag,str(obj)],text=True)
        if actual!=(a.build/"codegen"/filename).read_text():
            raise AssertionError("saved disassembly/resources are not from this exact object")
    native=kernels(a.build/"codegen/image.sass")
    resources={k:v.strip() for k,v in re.findall(r"Function (\S+):\n([^\n]+)",
               (a.build/"codegen/resources.txt").read_text())}
    verify(native,resources,authority)
    if a.self_test:
        name=next(iter(native))
        plants=[({k:v for k,v in native.items() if k!=name},resources),
                (native|{name:[native[name][0]+" WRONG",*native[name][1:]]},resources),
                (native|{name:list(reversed(native[name]))},resources),
                (native,resources|{name:resources[name]+" WRONG"})]
        for bad,regs in plants:
            try: verify(bad,regs,authority)
            except AssertionError: pass
            else: raise AssertionError("missing body/operand/order/resource plant escaped")
    print(json.dumps(dict(configuration=receipt["configuration"],status="PASS",
        source_anchor=authority["source_revision"],bodies=4,resources="IDENTICAL",
        instruction_sites=authority["instruction_sites"],
        native_sha256=authority["native_sha256"],negatives=4 if a.self_test else 0,
        scope="STATIC_NATIVE_IDENTITY_NOT_NEW_PERFORMANCE")))


if __name__=="__main__": main()
