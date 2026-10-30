"""Native ZW890/PPU1.5 forward. No PPU1.0 or Hopper fallback."""
from .backends.loading import load_ppu15


def gdn_chunk_ppu15(q, k, v, g, beta, initial_state=None, output_final_state=True, *, delivery="native"):
    if delivery != "native":
        raise ValueError("ppu15 owns its native layouts; PPU1.0 delivery selectors do not apply")
    inputs = tuple(x.contiguous() for x in (q, k, v, g, beta))
    initial = initial_state.contiguous() if initial_state is not None else None
    output, final = load_ppu15().forward(*inputs, initial, output_final_state)
    return output, final if output_final_state else None
