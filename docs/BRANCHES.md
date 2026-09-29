# Main and archived experiments

`main` is the only development branch. It contains the retained PPU1.0
V32/8 implementation alongside the independent NVIDIA SM80/SM90 backends.
See [PPU usage](PPU_RETAINED_PATH.md) and [SM90 installation](SM90_INSTALL.md).

Unselected experiments and their measurement records are preserved, not
maintained as parallel development branches:

- `archive/ppu10-before-main-20260929`: complete former tuning branch,
  including all geometry experiments and evidence.
- `archive/ppu10-tuning-closed-20260929`: the tuning closure before branch
  consolidation.
- Other `archive/ppu10-*` tags preserve individual experiment checkpoints.

The marginal V16/4 per-shape selector remains unadopted. Archiving a candidate
does not promote it, erase its result, or imply it was incorrect.

Use the mainline in existing checkouts:

```bash
git fetch origin --prune --tags
git switch main
git pull --ff-only origin main
```

Historical code can be inspected without creating another branch:

```bash
git switch --detach archive/ppu10-before-main-20260929
```

Commit or stash local edits before switching checkouts; these commands do not
discard them. Native extensions built from an old branch must be rebuilt when
switching to main.
