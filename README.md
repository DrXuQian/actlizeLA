# actlizeLA v0.2.2 prebuilt assets

PPU1.0 / SDK2.2.1 only. Source release: `6505987ca68fcba7c1a530b054c96388b22b8c01`.

This branch holds LFS payloads for the GitHub Release publisher. It is not the
source branch. `assets/release-manifest.json` binds the exact source, wheel and
qualification files. The tag-triggered publisher verifies the complete package
against the source before creating the downloadable GitHub Release.

Host/ABI/install qualification is complete. Installed-wheel device/Flare/full-model
qualification remains NOT_RUN; see the attached notes and qualification JSON.
