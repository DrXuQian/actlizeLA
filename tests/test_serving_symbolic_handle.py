"""Fake/Dynamo tracing must keep process-local native handles symbolic."""

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv

from actlize_la.serving._native import PreparedNative, _plans, _serving_out


def test_unbacked_handle_passes_real_fake_dispatch_without_native_loading():
    shape_env = ShapeEnv()
    with FakeTensorMode(shape_env=shape_env):
        handle = shape_env.create_unbacked_symint()
        inputs = [torch.empty(7, 24, 256, dtype=torch.bfloat16, device="cuda:0")]
        buffers = [torch.empty_like(inputs[0]), torch.empty(2, dtype=torch.int32, device="cuda:0")]
        assert _serving_out(inputs, buffers, handle) is None
        assert not shape_env.guards


def test_symbolic_handle_survives_fullgraph_dynamo_tracing():
    graphs = []

    def backend(module, examples):
        graphs.append(module)
        # Only the real compiler boundary is exercised; no native launch is made.
        return lambda *args: ()

    def forward(source, output, handle):
        _serving_out([source], [output], handle.item())

    with torch._dynamo.config.patch(capture_scalar_outputs=True):
        torch.compile(forward, backend=backend, fullgraph=True, dynamic=False)(
            torch.empty(7, 128), torch.empty(7, 128), torch.tensor(112233)
        )
    assert len(graphs) == 1
    calls = [n for n in graphs[0].graph.nodes if n.target is torch.ops.actlize_la.serving_out.default]
    assert len(calls) == 1
    assert isinstance(calls[0].args[-1], torch.fx.Node)


def test_concrete_handle_still_checks_retained_owner_metadata():
    owner = PreparedNative.__new__(PreparedNative)
    owner.device = torch.device("cpu")
    owner.inputs, owner.buffers = [torch.empty(2, 3)], [torch.empty(2, 3)]
    owner.signatures = [(tuple(t.shape), t.dtype, t.stride()[1:], t.is_contiguous())
                        for t in (*owner.inputs, *owner.buffers)]
    handle = 9988776655
    _plans[handle] = owner
    try:
        with FakeTensorMode():
            assert _serving_out([torch.empty(2, 3)], [torch.empty(2, 3)], handle) is None
            with pytest.raises(ValueError, match="shape/dtype/inner layout"):
                _serving_out([torch.empty(2, 4)], [torch.empty(2, 3)], handle)
    finally:
        del _plans[handle]


def test_real_dispatch_still_rejects_an_expired_handle():
    with pytest.raises(RuntimeError, match="serving plan expired"):
        _serving_out([torch.empty(1)], [torch.empty(1)], 9988776655)
