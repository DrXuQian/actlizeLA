"""Preparation-time PPU1.0 route inspection; never called from forward/replay."""
from importlib import import_module
from pathlib import Path

from .gdn_interface import _resolve_forward
from .backends.loading import loaded_image_info, _file_image


def _metadata(tensor):
    if tensor is None:
        return None
    return dict(shape=list(tensor.shape), stride=list(tensor.stride()),
                dtype=str(tensor.dtype), device=str(tensor.device))


def get_gdn_route_info(q, k, v, g, beta, initial_state=None, output_final_state=True,
                       *, load_binary=True, **forward_options):
    """Describe the same route as gdn_forward without executing a device kernel.

    Call before timing/capture, with precisely the forward call's arguments.
    load_binary=False reports policy only, not an executable or device proof.
    The default loads that route's binding, checks its entry and records both
    the binding and actual mapped kernel DSO. File changes after load fail.
    This inspector currently covers PPU1.0; other targets are explicit errors.
    """
    route = _resolve_forward(q, initial_state, **forward_options)
    if route["backend"] != "ppu10":
        raise ValueError("get_gdn_route_info currently requires backend='ppu10'")
    module = import_module(f".{route['module']}", __package__)
    if route["algorithm"] == "residual":
        try:
            native_entry = module.RESIDUAL_ENTRYPOINTS[route["delivery"]]
        except KeyError:
            raise ValueError(f"unknown residual delivery: {route['delivery']}") from None
    elif route["algorithm"] == "wy":
        if route["delivery"] not in module.DELIVERIES:
            raise ValueError(f"unknown WY delivery: {route['delivery']}")
        native_entry = "forward"
    else:
        native_entry = "forward_gdn_chunk_auto"
    result = dict(
        schema=1, api="gdn_forward", backend=route["backend"], algorithm=route["algorithm"],
        delivery=route["delivery"], python_entry=f"actlize_la.{route['module']}.{route['entry']}",
        python_sources_on_disk=[_file_image(path) for path in (
            _resolve_forward.__code__.co_filename, module.__file__,
            Path(__file__).with_name("backends") / "targets.json")],
        native_entry=native_entry, numerical_contract=route["numerical_contract"],
        state_layout="B,Hv,K,V", final_state_dtype=route["final_state_dtype"],
        output_final_state=bool(output_final_state),
        inputs={name: _metadata(value) for name, value in
                zip(("q", "k", "v", "g", "beta", "initial_state"),
                    (q, k, v, g, beta, initial_state))},
        native_status="NOT_LOADED", device_execution="NOT_RUN",
    )
    if route["algorithm"] == "original":
        result["internal_scan_reset_selection"] = "DATA_DEPENDENT_NOT_OBSERVED"
    if load_binary:
        native = module._backend()
        if not callable(getattr(native, native_entry, None)):
            raise RuntimeError(f"selected native entry is missing: {native_entry}; no fallback")
        from ._gdn_ppu10 import PPU10Backend
        if isinstance(native, PPU10Backend):
            result["native_images"] = native.image_info()
            result["native_entry"] = "actlize_la_ppu10_forward_prepared"
            result["native_provider"] = native.provider
            result["manifest_sha256"] = native.manifest_sha256
        else:
            dependency = "libgdn_qsa_ppu.so" if route["algorithm"] == "original" else "libgdn_wy_ppu.so"
            result["native_images"] = loaded_image_info(native, dependency)
            result["native_provider"] = "explicit-extension"
        result["native_status"] = "LOADED_FILES_VERIFIED_NOT_DEVICE_ADMITTED"
    return result
