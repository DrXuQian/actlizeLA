#!/usr/bin/env python3
"""Prove host-only solve composition; do not infer a device performance win."""
import argparse
import hashlib
from pathlib import Path
import re
from host_disassembly import read_host_disassembly

from check_state_pipeline import block, code
from check_wy_binary import kernel_sequences

ROOT = Path(__file__).resolve().parents[2]
CONTROL = "gdn_wy_forward_residual_gate_cache"
SUBJECT = CONTROL + "_solve_static"
# Normalized original host body at actlizeLA 7616123, independently frozen
# before factoring. Removing just the two selection seams must restore it.
CONTROL_BODY_SHA256 = "2d21546b57b2264a2bbccc00f35a7fdd1a20bf83fe29a80910c556644d137963"
CONFIGURE = "if constexpr (StaticSolve) rc = solve_static::configure(); else rc = configure_split_prepare();"
LAUNCH = ("if constexpr (StaticSolve) rc = solve_static::launch_inverse(p, inverse_ws, stream); "
          "else rc = launch_split_inverse(p, inverse_ws, stream);")


def replace_once(text, old, new):
    old, new = code(old), code(new)
    if text.count(old) != 1:
        raise AssertionError("missing/ambiguous composition seam")
    return text.replace(old, new, 1)


def check_source(source, binding):
    text = code(source)
    body = block(text, "forward_gate_cache(")
    body = replace_once(body, "int rc;" + CONFIGURE, "int rc = configure_split_prepare();")
    body = replace_once(body, LAUNCH, "rc = launch_split_inverse(p, inverse_ws, stream);")
    body = body.replace("forward_gate_cache(", CONTROL + "(", 1)
    if hashlib.sha256(body.encode()).hexdigest() != CONTROL_BODY_SHA256:
        raise AssertionError("composition changed the frozen admission/workspace/state/output contract")
    args = "q,k,v,g,beta,initial,output,final,inverse,snapshots,vnew,gates,batch,sequence,q_heads,value_heads,gate_fp32,stream"
    for name, value in ((CONTROL, "false"), (SUBJECT, "true")):
        wrapper = block(text, name + "(")
        if wrapper[wrapper.index("{"):] != "{returnforward_gate_cache<" + value + ">(" + args + ");}":
            raise AssertionError("wrong ABI selector/argument forwarding: " + name)
    for token in (
        "Variant<=14&&(!Variant||Residual)",
        "Variant==12?gdn_wy_forward_residual_gate_cache:Variant==13?gdn_wy_forward_residual_gate_cache_solve_static:",
        'm.def("residual_gate_cache_solve_static",&forward<true,13>,',
    ):
        if code(binding).count(token) != 1:
            raise AssertionError("wrong composed Python/native binding")


def functions(disassembly):
    result, current = {}, None
    for line in disassembly.splitlines():
        match = re.match(r"^[0-9a-f]+ <(.+)>:$", line)
        if match:
            current = match[1]
            result[current] = []
        elif current:
            result[current].append(line)
    return result


def reachable_calls(bodies, entry):
    if entry not in bodies:
        raise AssertionError("linked host entry missing: " + entry)
    pending, visited, calls = [entry], set(), set()
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        for line in bodies[name]:
            match = re.search(r"\b(?:callq?|jmpq?)\s+[0-9a-f]+\s+<(.+)>", line)
            if not match:
                continue
            target = re.sub(r"\+0x[0-9a-f]+$", "", match[1])
            resolved = target.removesuffix("@plt")
            calls.add(resolved)
            # HGGC can expose its anonymous template via a weak PLT symbol.
            # Follow that exact composition helper, not other stage internals.
            if resolved in bodies and (not target.endswith("@plt") or
                                       "::forward_gate_cache<" in resolved):
                pending.append(resolved)
    return calls


def check_linked_host(disassembly):
    bodies = functions(disassembly)
    common = ("gate_cache::gdn_wy_residual_gate_cache_state(",
              "residual_warps8_hvlayout::launch_hvlayout_output(")
    regular = ("wy::configure_split_prepare(", "wy::launch_split_inverse(")
    static = ("solve_static::configure(", "solve_static::launch_inverse(")
    for entry, required, forbidden in ((CONTROL, regular, static), (SUBJECT, static, regular)):
        calls = reachable_calls(bodies, entry)
        for marker in (*required, *common):
            if not any(marker in name for name in calls):
                raise AssertionError(f"{entry}: missing actual host call {marker}")
        if any(marker in name for name in calls for marker in forbidden):
            raise AssertionError(f"{entry}: wrong solve reachable from actual host entry")
    return dict(entries=2, same_state_output=True, solve="REGULAR_vs_STATIC", device="NOT_RUN")


def compare_native(parent, candidate):
    old, new = kernel_sequences(parent), kernel_sequences(candidate)
    if len(old) != 37 or set(old) != set(new):
        raise AssertionError("composition must preserve all37 native kernels, adding none")
    for name, ops in old.items():
        if new[name] != ops:
            raise AssertionError("native operand/instruction sequence changed: " + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--library", type=Path)
    parser.add_argument("--parent-isa", type=Path)
    parser.add_argument("--isa", type=Path)
    args = parser.parse_args()
    source = (ROOT / "csrc/gdn_chunk/gdn_wy_residual_gate_cache_ppu.cu").read_text()
    binding = (ROOT / "csrc/gdn_chunk/gdn_wy_ops.cpp").read_text()
    check_source(source, binding)
    if args.self_test:
        for label, before, after in (
            ("old-solve", "forward_gate_cache<true>", "forward_gate_cache<false>"),
            ("wrong-configure", "rc = solve_static::configure()", "rc = configure_split_prepare()"),
            ("inverse-alias", "inverse_ws.snapshots = ws.w", "inverse_ws.snapshots = ws.snapshots"),
            ("wrong-state", "gdn_wy_residual_gate_cache_state<<<", "gdn_wy_residual_warps8_hvlayout_state<<<"),
            ("wrong-output", "return launch_hvlayout_output(", "return launch_hlayout_output("),
        ):
            if source.count(before) != 1:
                raise AssertionError("ambiguous source negative: " + label)
            try:
                check_source(source.replace(before, after, 1), binding)
            except AssertionError:
                print("[gate-cache solve negative]", label, "EXPECTED-RED/PASS")
            else:
                raise AssertionError("escaped source negative: " + label)
        try:
            check_source(source, binding.replace("&forward<true, 13>", "&forward<true, 12>"))
        except AssertionError:
            print("[gate-cache solve negative] wrong-binding EXPECTED-RED/PASS")
        else:
            raise AssertionError("escaped wrong-binding negative")
    if args.library:
        host = read_host_disassembly(args.library)
        print("[gate-cache solve host]", check_linked_host(host))
        if args.self_test:
            for before, after in (("solve_static::launch_inverse(", "launch_split_inverse("),
                                  (SUBJECT + ">:", "MISSING_COMPOSED_ENTRY>:")):
                if before not in host:
                    raise AssertionError("missing host negative seam")
                try:
                    check_linked_host(host.replace(before, after))
                except AssertionError:
                    print("[gate-cache solve linked negative] EXPECTED-RED/PASS", before)
                else:
                    raise AssertionError("wrong actual linked solve escaped")
    if bool(args.parent_isa) != bool(args.isa):
        parser.error("parent-isa and isa must be supplied together")
    if args.isa:
        compare_native(args.parent_isa.read_text(), args.isa.read_text())
        print("[gate-cache solve] native37/37=IDENTICAL new-device-kernels=0")
    print("[gate-cache solve] PASS scope=LOCAL_COMPOSITION_NOT_DEVICE_PERFORMANCE")


if __name__ == "__main__":
    main()
