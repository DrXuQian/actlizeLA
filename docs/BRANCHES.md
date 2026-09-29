# Branch organization

Updated 2026-09-29. This is reference cleanup, not kernel promotion or a
change to the existing default/public dispatch.

## Retained remote branches

| Branch | Purpose |
|---|---|
| `main` | Existing mainline, unchanged at `6f6a2b7` by this cleanup |
| `agent/ppu10-geometry-20260928` | Complete PPU1.0 tuning history and closure; all measured controls, opt-in candidates and evidence remain here |

The current tuning branch name is intentionally retained so existing box
checkouts can continue to pull. The development chain is not merged into
`main` merely to reduce the number of branches. In particular, the user
declined the marginal V16/4 shape-based selector; default routing is unchanged.

## Archived experiment tips

Each retired remote branch had **zero commits not reachable from the retained
tuning branch**. Tags were created and the seven branch refs removed in one
atomic operation, with exact old-SHA leases guarding concurrent updates.
No commit history or worktree contents were deleted.

| Deleted branch | Recovery tag | Original tip |
|---|---|---|
|`agent/ppu10-solve-gate-cache-20260928`|`archive/ppu10-solve-gate-cache-20260928`|`c99d84bb93e32849a9eeac312809c3a4072a6c56`|
|`agent/ppu10-full-chunk-state-20260928`|`archive/ppu10-full-chunk-state-20260928`|`e379fea12adb07a3d8917b4b730bcfee10e0649d`|
|`agent/ppu10-full-chunk-stages-20260928`|`archive/ppu10-full-chunk-stages-20260928`|`8bbd3dc220367933add1cc5d606e0fa9e54231a0`|
|`agent/ppu10-first-chunk-20260928`|`archive/ppu10-first-chunk-20260928`|`f967cfe87251c7559f07bb30a39c49e4163a9d1d`|
|`agent/ppu10-mixed-tail-20260928`|`archive/ppu10-mixed-tail-20260928`|`2bde8331b05471293dd834cf51d57ac7e21fd496`|
|`agent/ppu10-inverse-register-20260928`|`archive/ppu10-inverse-register-20260928`|`e1627d0968604c6fde9d7273824abe3850b99b04`|
|`agent/ppu10-paired-conversion-20260928`|`archive/ppu10-paired-conversion-20260928`|`8777067bb4785bd36de1f4991bd0029513895818`|

`archive/ppu10-tuning-closed-20260929` preserves the exact tuning closure
commit `643e8359215a551ebf93fe410b2e2230eb0ff77d`, before this branch-only
documentation. Historical experiment instructions naming retired branches
refer to these archived snapshots, not additional active development lines.

Refresh an existing checkout:

```bash
git fetch origin --prune --tags
```

To inspect an old experiment without moving an existing branch, for example:

```bash
git switch -c inspect/first-chunk archive/ppu10-first-chunk-20260928
```

The old pre-history-rewrite local `main` (`481fc33`) was preserved locally
as `archive/legacy-main-20260817`, then local `main` was recreated to track
the current remote mainline. The already-deleted legacy remote experiment
tip `545ea0b` was preserved as local tag
`archive/local-legacy-all-aiu-forward-20260821` before pruning its stale
tracking ref. These legacy snapshots were not republished as remote branches.
