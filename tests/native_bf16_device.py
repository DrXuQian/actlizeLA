#!/usr/bin/env python3
"""Preflight only: verify native BF16 scalar semantics before any ACU capture."""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def corpus():
    hi = np.arange(65536, dtype=np.uint32)[:, None] << 16
    lo = np.array([0, 1, 0x7fff, 0x8000, 0x8001, 0xffff], dtype=np.uint32)
    edge = (hi | lo).reshape(-1)
    random = np.random.default_rng(0xBF16).integers(0, 2**32, size=65536, dtype=np.uint32)
    return np.concatenate((edge, random))


def oracle(bits):
    """Integer RNE definition, independent of cutlass's round/sticky branch."""
    high, low = bits >> 16, bits & 0xffff
    carry = (low > 0x8000) | ((low == 0x8000) & ((high & 1) != 0))
    result = ((high + carry) & 0xffff).astype(np.uint16)
    nan = ((bits & 0x7f800000) == 0x7f800000) & ((bits & 0x7fffff) != 0)
    result[nan] = 0x7fff  # retained cutlass constructor's signed/payload-independent NaN contract
    return result


def require_bits(got, want):
    if got.shape != want.shape or got.dtype != np.uint16 or want.dtype != np.uint16:
        raise AssertionError('BF16 bit-test geometry/dtype changed')
    bad = np.flatnonzero(got != want)
    if len(bad):
        i = int(bad[0])
        raise AssertionError(f'BF16 raw mismatch {len(bad)}/{len(want)}, first={i}, got=0x{got[i]:04x}, want=0x{want[i]:04x}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    result = dict(status='FAIL', scope='RAW-BIT conversion preflight; no timing admission')
    try:
        import torch
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from actlize_la.serving import NativeLibrary
        from actlize_la.serving.device import current_device_identity
        if not torch.cuda.is_available():
            result.update(status='SKIP', reason='no device; native conversion NOT_RUN')
            return 3
        data = corpus()
        want = oracle(data)
        inputs = torch.from_numpy(data.view(np.int32).copy()).cuda()
        old = torch.empty(len(data), dtype=torch.int16, device='cuda')
        new = torch.empty_like(old)
        lib = NativeLibrary(args.library, target='ppu10')
        fn = lib.lib.actlize_la_ppu10_native_bf16_probe
        fn.argtypes, fn.restype = [C.c_void_p] * 3 + [C.c_int, C.c_void_p], C.c_int
        for _ in range(2):
            rc = fn(inputs.data_ptr(), old.data_ptr(), new.data_ptr(), len(data), torch.cuda.current_stream().cuda_stream)
            if rc: raise RuntimeError(f'conversion probe launch status={rc}')
            torch.cuda.synchronize()
            require_bits(old.cpu().numpy().view(np.uint16), want)
            require_bits(new.cpu().numpy().view(np.uint16), want)
        broken = want.copy(); broken[0] ^= np.uint16(1)
        try: require_bits(broken, want)
        except AssertionError: pass
        else: raise AssertionError('one-bit negative escaped')
        result.update(status='PASS', count=len(data), repeats=2,
                      input_sha256=hashlib.sha256(data.tobytes()).hexdigest(),
                      expected_sha256=hashlib.sha256(want.tobytes()).hexdigest(),
                      library_sha256=hashlib.sha256(args.library.read_bytes()).hexdigest(),
                      device=current_device_identity('ppu10'), negative='one-bit EXPECTED_RED')
        print('[native BF16] PASS: all high16 patterns x6 low16 boundaries +65536 random; signed-zero/subnormal/Inf/NaN; raw-bit')
        return 0
    except Exception as error:
        result['error'] = repr(error)
        raise
    finally:
        args.receipt.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__': sys.exit(main())
