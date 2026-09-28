# One upload archive for the full-stage experiment

This is a packaging update, not a change to the frozen experiment criteria
in PPU10_FULL_CHUNK_STAGES.md. Kernel code, selected shapes and timing rules
are unchanged.

New runs of tools/run_ppu10_full_chunk_stages_box.sh automatically finish by
creating **full-chunk-stages.tar.gz** in their run directory. Upload only this
file. It contains the six original capture archives plus measurement SHA,
binary hashes, numerical admission and an outer checksum manifest.

An already completed run needs no rebuild or reprofile:

~~~bash
bash tools/pack_ppu10_full_chunk_stages.sh /workspace/EXISTING_RUN_DIRECTORY
~~~

The packer preserves the measurement's original SHA even if the checkout now
contains this newer script. Missing/empty captures fail before packaging;
an existing upload archive is never overwritten. A package is not a numerical
or performance PASS: each enclosed capture retains its independent admission,
identity and checksum evidence for analysis.
