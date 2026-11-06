#!/usr/bin/env python3
"""Preparation-time, data-only native binary descriptor. Never used by run()."""

import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
from actlize_la.serving import NativeLibrary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument(
        "--target", choices=("ppu10", "ppu15", "cuda_sm120"), required=True
    )
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lib = NativeLibrary(args.library, target=args.target)
    symbols = (
        "ala_pointwise_v1_run",
        "ala_shortconv_v1_run",
        "ala_ple_hash_v1_run",
        "ala_gdn_verify_v2",
        "ala_gdn_commit_v2",
        "ala_gdn_checked_v2",
        "ala_prefill_checked_v2",
        "ala_gdn_run",
        "ala_gdn_commit",
        "ala_prefill_run",
        "ala_prefill_configure",
        "ala_qsa_run",
        "ala_conv_capture",
        "ala_conv_rollback_zero",
    )
    for symbol in symbols:
        getattr(lib.lib, symbol)
    sizes = lib.lib.ala_aux_v1_struct_size
    sizes.argtypes = [C.c_int]
    sizes.restype = C.c_int
    result = dict(
        schema="actlize-la-native-serving-v1",
        source_sha=args.source_sha,
        target=args.target,
        library=str(args.library.resolve()),
        sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
        launch_symbols=list(symbols),
        aux_struct_bytes={
            n: sizes(i)
            for i, n in enumerate(
                ("tensor", "rows", "pointwise", "shortconv", "ple_hash")
            )
        },
        legacy_abi=1,
        aux_abi=1,
        gdn_transaction_abi=2,
        ownership="caller retains module, context, stream, graph and all referenced buffers until execution drains",
        execution="AOT-native launch; graph replay contains no Python callbacks",
        validation="BINARY_METADATA_ONLY_NOT_DEVICE_OR_MODEL_ADMISSION",
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
