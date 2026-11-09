"""Host contracts only: no CPU stand-in is counted as PPU numerical evidence."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from actlize_la.serving import PreparedGDNCore, gdn_core_workspace_specs
from actlize_la.serving import gdn_core
from actlize_la.serving._native import PreparedNative, reject_aliases


def operands(initial=True, final=True, fp32_gate=True):
    b, t, hk, hv = 2, 65, 2, 6
    q = torch.empty(b, t, hk, 128, dtype=torch.bfloat16)
    v = torch.empty(b, t, hv, 128, dtype=torch.bfloat16)
    shape = (b, t, hv)
    return dict(q=q, k=torch.empty_like(q), v=v,
                g=torch.empty(shape, dtype=torch.float32 if fp32_gate else torch.bfloat16),
                beta=torch.empty(shape, dtype=torch.bfloat16), output=torch.empty_like(v),
                initial_state=torch.empty(b, hv, 128, 128) if initial else None,
                final_state=torch.empty(b, hv, 128, 128) if final else None,
                workspace={n: torch.empty(s, dtype=d) for n, (s, d)
                           in gdn_core_workspace_specs(b, t, hk, hv).items()})


@pytest.fixture
def transport(monkeypatch):
    # Deliberately replace only the CUDA owner setup, never perform numerical
    # computation. The real geometry, alias and ctypes argument assembly run.
    def bind(self, library, inputs, buffers):
        reject_aliases(inputs, buffers)
        self.library, self.inputs, self.buffers = library, inputs, buffers
        self.device = torch.device("cpu")

    monkeypatch.setattr(PreparedNative, "bind", bind)
    monkeypatch.setattr(PreparedNative, "stream", lambda self: 123456)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    return SimpleNamespace(target="ppu10", lib=SimpleNamespace(
        actlize_la_ppu10_configure=Mock(return_value=0),
        actlize_la_ppu10_forward_prepared=Mock(return_value=0)))


@pytest.mark.parametrize("initial", [False, True])
@pytest.mark.parametrize("final", [False, True])
@pytest.mark.parametrize("fp32", [False, True])
def test_real_wire_order_and_nulls(transport, initial, final, fp32):
    values = operands(initial, final, fp32)
    plan = PreparedGDNCore(transport, **values)
    for _ in range(8):
        plan.launch(plan.inputs, plan.buffers)
    args = transport.lib.actlize_la_ppu10_forward_prepared.call_args.args
    ptr = lambda name: values[name].data_ptr() if values[name] is not None else None
    assert args[:8] == tuple(ptr(n) for n in
                            ("q", "k", "v", "g", "beta", "initial_state", "output", "final_state"))
    assert args[8:12] == tuple(values["workspace"][n].data_ptr() for n in
                              ("inverse", "snapshots", "vnew", "gates"))
    assert args[12:] == (2, 65, 2, 6, fp32, 123456)
    assert values["workspace"]["inverse"].numel() == 2 * 6 * 2 * 16384
    transport.lib.actlize_la_ppu10_configure.assert_called_once_with()
    PreparedGDNCore(transport, **values)
    transport.lib.actlize_la_ppu10_configure.assert_called_once_with()


@pytest.mark.parametrize("plant", ["inverse", "output-alias", "state-alias", "gate", "key", "state-dtype", "state-layout", "missing", "extra", "target"])
def test_bad_operands_rejected_before_config_or_launch(transport, plant):
    values = operands()
    if plant == "inverse":
        values["workspace"]["inverse"] = torch.empty(24, 64, 64, dtype=torch.bfloat16)
    elif plant == "output-alias":
        values["output"] = values["v"]
    elif plant == "state-alias":
        values["final_state"] = values["initial_state"]
    elif plant == "gate":
        values["g"] = values["g"].half()
    elif plant == "key":
        values["k"] = values["k"][:, :, :1].contiguous()
    elif plant == "state-dtype":
        values["initial_state"] = values["initial_state"].bfloat16()
    elif plant == "state-layout":
        values["initial_state"] = values["initial_state"].transpose(-1, -2)
    elif plant == "missing":
        del values["workspace"]["gates"]
    elif plant == "extra":
        values["workspace"]["extra"] = torch.empty(1)
    elif plant == "target":
        transport.target = "ppu15"
    with pytest.raises(ValueError):
        PreparedGDNCore(transport, **values)
    transport.lib.actlize_la_ppu10_configure.assert_not_called()
    transport.lib.actlize_la_ppu10_forward_prepared.assert_not_called()


def test_capture_cannot_configure_cold_library(transport, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="before capture"):
        PreparedGDNCore(transport, **operands())
    transport.lib.actlize_la_ppu10_configure.assert_not_called()


@pytest.mark.parametrize('selected,name', [(0, 'retained'), (1, 'solve8'), (2, 'solve8-value64-warps8')])
def test_core_route_is_native_host_query(transport, selected, name):
    transport.lib.actlize_la_ppu10_core_route = Mock(return_value=selected)
    route = gdn_core.gdn_core_route(transport, 1, 2048, 16, 32)
    assert route == {'id': selected, 'name': name, 'entry': 'actlize_la_ppu10_forward_prepared'}
    transport.lib.actlize_la_ppu10_core_route.assert_called_once_with(1, 2048, 16, 32)
    transport.lib.actlize_la_ppu10_forward_prepared.assert_not_called()
    transport.lib.actlize_la_ppu10_configure.assert_not_called()


def test_unknown_core_route_fails_closed(transport):
    transport.lib.actlize_la_ppu10_core_route = Mock(return_value=3)
    with pytest.raises(RuntimeError, match='unknown native GDN core route'):
        gdn_core.gdn_core_route(transport, 2, 8192, 16, 48)


def test_legacy_library_query_is_backward_compatible(transport):
    plan = PreparedGDNCore(transport, **operands())
    assert plan.core_route['id'] == 0 and plan.core_route['name'] == 'retained'


def test_launch_failure_remains_failure(transport):
    plan = PreparedGDNCore(transport, **operands())
    transport.lib.actlize_la_ppu10_forward_prepared.return_value = 7
    with pytest.raises(RuntimeError, match="status=7"):
        plan.launch(plan.inputs, plan.buffers)


@pytest.mark.parametrize("shape", [(0, 64, 1, 2), (1, 2**31, 1, 2), (2**30, 64, 1, 2), (1, 64, 2, 3)])
def test_bad_geometry_does_not_allocate(shape):
    with pytest.raises(ValueError):
        gdn_core_workspace_specs(*shape)


def test_default_packaged_and_explicit_extension_selection(monkeypatch):
    from actlize_la import _gdn_ppu10, gdn_residual_interface as residual
    packaged, explicit = object(), object()
    load = Mock(return_value=packaged)
    old = Mock(return_value=explicit)
    monkeypatch.setattr(_gdn_ppu10, "packaged_backend", load)
    monkeypatch.setattr(residual, "_legacy_backend", old)
    monkeypatch.delenv("GDN_QSA_WY_EXTENSION", raising=False)
    monkeypatch.delenv("ACTLIZE_LA_SERVING_LIBRARY", raising=False)
    assert residual._backend() is packaged
    old.assert_not_called()
    monkeypatch.setenv("GDN_QSA_WY_EXTENSION", "/explicit/old.so")
    assert residual._backend() is explicit
    old.side_effect = RuntimeError("bad explicitly selected image")
    with pytest.raises(RuntimeError, match="bad explicitly"):
        residual._backend()
    assert load.call_count == 1


def test_explicit_source_serving_library_and_conflicting_env(monkeypatch):
    from actlize_la import _gdn_ppu10, gdn_residual_interface as residual
    load = Mock(return_value=object())
    monkeypatch.setattr(_gdn_ppu10, "explicit_backend", load)
    monkeypatch.delenv("GDN_QSA_WY_EXTENSION", raising=False)
    monkeypatch.setenv("ACTLIZE_LA_SERVING_LIBRARY", "/built/serving.so")
    assert residual._backend() is load.return_value
    load.assert_called_once_with("/built/serving.so", "")
    monkeypatch.setenv("GDN_QSA_WY_EXTENSION", "/old/extension.so")
    with pytest.raises(ValueError, match="only one"):
        residual._backend()
    assert load.call_count == 1


def test_same_kernel_output_only_allocating_adapter(monkeypatch):
    # CPU transport stand-in exercises the allocating wrapper's wiring only.
    from actlize_la import _gdn_ppu10
    backend = object.__new__(_gdn_ppu10.PPU10Backend)
    backend.library = object()
    values = operands()
    class Input:
        is_cuda, ndim = True, 4
        shape, device = values["q"].shape, torch.device("cpu")
    monkeypatch.setattr(gdn_core, "_geometry", lambda *a: (2, 65, 2, 6))
    plan = Mock()
    construct = Mock(return_value=plan)
    monkeypatch.setattr(gdn_core, "PreparedGDNCore", construct)
    output, final = backend.residual_full_chunk(Input(), values["k"], values["v"],
        values["g"], values["beta"], values["initial_state"], False)
    assert final is None and output.shape == values["v"].shape
    assert construct.call_args.kwargs["initial_state"] is values["initial_state"]
    assert construct.call_args.kwargs["final_state"] is None
    plan.assert_called_once_with()
