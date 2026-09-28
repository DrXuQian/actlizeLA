#!/usr/bin/env python3
"""Pin the optimization structure to the actual upstream kernels, not a model.

Control expressions, launch geometries, barrier/wait cadence and dispatch
statements must remain upstream-identical. Target primitive implementations
are proved separately by l006 and the hgcc binary gate.
"""
import argparse
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = "aa04271"
DEVICE = (
    "gdn_kernel.cu", "gdn_scan_stage1.cu", "gdn_scan_stage1_reset.cu",
    "gdn_scan_stage2.cu", "gdn_scan_stage2_blelloch.cu", "gdn_scan_stage3_reset.cu",
)


def code(text):
    text = re.sub(r"//[^\n]*|/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"\s+", "", text)


def controls(text):
    text = re.sub(r"//[^\n]*|/\*.*?\*/", "", text, flags=re.S)
    found = []
    for match in re.finditer(r"\b(for|while|if|switch)\s*(?:constexpr\s*)?\(", text):
        start, depth, end = match.end(), 1, match.end()
        while depth:
            depth += (text[end] == "(") - (text[end] == ")")
            end += 1
        found.append((match.group(1), code(text[start:end - 1])))
    return found


def compare(old, new):
    errors = []
    if controls(old) != controls(new):
        errors.append("control-flow")
    if re.findall(r"<<<(.*?)>>>", code(old)) != re.findall(r"<<<(.*?)>>>", code(new)):
        errors.append("launch-geometry")
    for token in ("__syncthreads()", "__syncwarp()", "cp_async_fence()",
                  "cp_async_wait<0>()", "cp_async_wait<1>()", "gemm(thr_mma,",
                  "RawStage raw[2]", "InStage in[2]", "s_reg[kCPW][K_BLOCKS_TOTAL]"):
        if code(old).count(code(token)) != code(new).count(code(token)):
            errors.append(f"cadence/lifetime:{token}")
    if "SFragT s_reg" in old:
        # Merely counting the declaration would allow accidentally moving
        # persistent state inside the chunk loop. Pin the scope as well.
        body = code(new[new.index("gdn_recurrence_fused_kernel"):])
        declaration = "SFragTs_reg[kCPW][K_BLOCKS_TOTAL];"
        loop = "for(intt=t0;t<t1;++t){"
        if declaration not in body or loop not in body or body.index(declaration) > body.index(loop):
            errors.append("register-state-not-hoisted")
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    pairs = {}
    total_controls = 0
    for name in (*DEVICE, "gdn_ops.cu"):
        path = f"csrc/gdn_chunk/{name}"
        old = subprocess.check_output(["git", "show", f"{UPSTREAM}:{path}"],
                                      cwd=ROOT, text=True)
        new = (ROOT / path).read_text()
        errors = compare(old, new)
        if name == "gdn_ops.cu":
            # Host arithmetic, reset thresholds and all dispatch calls are
            # exactly upstream code. Only includes / opaque type declaration
            # above the first public launcher differ.
            if code(old[old.index('extern "C"'):]) != code(new[new.index('extern "C"'):]):
                errors.append("host-dispatch")
        if errors:
            raise AssertionError(f"{name}: {errors}")
        total_controls += len(controls(old))
        pairs[name] = (old, new)
    if args.self_test:
        old, new = pairs["gdn_kernel.cu"]
        for label, token, replacement in (
            ("remove-prefetch-wait", "cp_async_wait<1>();", "cp_async_wait<0>();"),
            ("remove-second-buffer", "RawStage raw[2]", "RawStage raw[1]"),
            ("break-loop", "for (int k = 0;", "for (int k = 1;"),
        ):
            if token not in new:
                raise AssertionError(f"negative plant not applied: {label}")
            if not compare(old, new.replace(token, replacement, 1)):
                raise AssertionError(f"negative escaped: {label}")
            print(f"[PPU structure negative] {label} EXPECTED-RED/PASS")
        declaration = "SFragT s_reg[kCPW][K_BLOCKS_TOTAL];"
        start = new.index("gdn_recurrence_fused_kernel")
        prefix, body = new[:start], new[start:]
        moved = body.replace(declaration, "", 1).replace(
            "for (int t = t0; t < t1; ++t) {",
            "for (int t = t0; t < t1; ++t) { " + declaration, 1)
        if "register-state-not-hoisted" not in compare(old, prefix + moved):
            raise AssertionError("state-lifetime negative escaped")
        print("[PPU structure negative] state-inside-loop EXPECTED-RED/PASS")
    print(f"[PPU original structure] PASS source_TUs=7 controls={total_controls} "
          "host_dispatch=UPSTREAM-IDENTICAL reset+register-replay+double-buffer+scan=PRESERVED")


if __name__ == "__main__":
    main()
