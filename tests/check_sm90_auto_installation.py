"""Installed-frontend/native-bundle integration without executing GPU code.

Run outside the source directory after installation. For a relocated wheel,
set ACTLIZE_LA_SM90_BUNDLE to the separately built complete bundle.
"""
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

from actlize_la import gdn_forward, load_sm90
from actlize_la.sm90_policy import DeviceProfile


def main():
    automatic = load_sm90()
    labels = ("value64", "value64-local-inverse", "value128-paired")
    ops = automatic._variants
    assert set(ops) == set(labels)
    assert len({id(op._module) for op in ops.values()}) == 3
    assert len({op._module.__name__ for op in ops.values()}) == 3
    with ExitStack() as stack:
        # Real binaries, real configuration labels, real shape dispatch.
        # Only the last native call and device discovery are replaced: no GPU.
        stack.enter_context(patch("actlize_la.sm90_policy.device_profile",
                                  return_value=DeviceProfile("NVIDIA H800 PCIe", 114, (9, 0))))
        native = {name: stack.enter_context(patch.object(op._module, "forward",
                  Mock(return_value=(name, "state")))) for name, op in ops.items()}
        for b, t, expected in ((1, 2048, labels[0]), (1, 8192, labels[1]), (2, 2048, labels[2])):
            device = SimpleNamespace(type="cuda", index=0)
            q = SimpleNamespace(ndim=4, shape=(b, t, 16, 128), device=device)
            v = SimpleNamespace(ndim=4, shape=(b, t, 32, 128), device=device)
            k, g, beta, h0 = object(), object(), object(), object()
            assert gdn_forward(q, k, v, g, beta, initial_state=h0) == (expected, "state")
            native[expected].assert_called_once_with(q, k, v, g, beta, h0, True)
    print("[SM90 auto install] PASS real-native-modules=3 public-auto-routes=3 "
          "scope=IMPORT+DISPATCH_ONLY device=NOT_RUN")


if __name__ == "__main__":
    main()
