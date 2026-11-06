"""Complete native contract checks, only at explicit library preparation."""

import ctypes as C

from .._serving_contract import ABI_VERSION, EXPORTS, TARGET_IDS


def inspect_abi(lib, target):
    # Late imports avoid cycles with the prepared owners. No tensor/device work.
    from .auxiliary import HashV1, PointwiseV1, RowsV1, ShortConvV1, TensorV1
    from .conv_history import _Conv
    from .gdn import _Commit, _Gdn, _Prefill
    from .gdn_transaction import TransactionV2
    from .qsa import _Qsa

    for name in ("ala_serving_version", "ala_serving_target"):
        getattr(lib, name).argtypes = []
        getattr(lib, name).restype = C.c_int
    version, numeric = lib.ala_serving_version(), lib.ala_serving_target()
    if version != ABI_VERSION or numeric != TARGET_IDS[target]:
        raise ValueError(
            f"serving ABI/target mismatch: version={version}, target={numeric}; no binary fallback"
        )
    for name in EXPORTS:
        getattr(lib, name)  # missing symbols are fatal, never a partial backend
    sizes = {}
    groups = (
        ("ala_serving_struct_size", (_Gdn, _Commit, _Prefill, _Conv)),
        (
            "ala_aux_v1_struct_size",
            (TensorV1, RowsV1, PointwiseV1, ShortConvV1, HashV1),
        ),
        ("ala_qsa_struct_size", (_Qsa,)),
        ("ala_gdn_transaction_v2_size", (TransactionV2,)),
    )
    for name, structs in groups:
        fn = getattr(lib, name)
        indexed = name in ("ala_serving_struct_size", "ala_aux_v1_struct_size")
        fn.argtypes = [C.c_int] if indexed else []
        fn.restype = C.c_int
        for index, struct in enumerate(structs):
            size = fn(index) if indexed else fn()
            if size != C.sizeof(struct):
                raise RuntimeError(
                    f"C/Python ABI size mismatch for {struct.__name__}: {size} != {C.sizeof(struct)}"
                )
            sizes[struct.__name__] = size
    return dict(
        version=version, target=numeric, exports=list(EXPORTS), struct_sizes=sizes
    )
