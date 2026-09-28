"""Prebuilt, shape-dispatched SM90 forward. No runtime compilation/autotuning."""
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path

from .sm90_policy import policy, policy_digest, selection_for

REGISTRATION_FILE = Path(__file__).with_name("_sm90_bundle.json")
BUILD_IDENTITY_FIELDS = ("compiler_sha256", "dependency_tree_sha256", "source_sha256", "torch")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class AutoSm90Forward:
    def __init__(self, variants):
        if set(variants) != set(policy()["configurations"]):
            raise ValueError("SM90 auto requires every registered candidate, no missing-arm fallback")
        if any(name != op.configuration for name, op in variants.items()):
            raise ValueError("SM90 bundle configuration mismatch")
        self._variants = dict(variants)

    def select(self, q, v, *, mode="device", sm_count=None):
        """Diagnostic metadata only; does not execute or time a kernel."""
        return selection_for(q, v, mode=mode, sm_count=sm_count)

    def __call__(self, q, k, v, g, beta, initial_state=None, output_final_state=True,
                 *, mode="device", sm_count=None):
        choice = self.select(q, v, mode=mode, sm_count=sm_count)
        return self._variants[choice.configuration](q, k, v, g, beta,
            initial_state=initial_state, output_final_state=output_final_state)


def load_bundle(directory):
    from .sm90 import load_sm90
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "bundle.json").read_text())
    if (manifest.get("schema") != 1 or manifest.get("target") != "cuda_sm90" or
            manifest.get("policy_sha256") != policy_digest()):
        raise ValueError("SM90 bundle target/policy identity mismatch; rebuild or reassemble")
    rows = manifest.get("configurations", {})
    if set(rows) != set(policy()["configurations"]):
        raise ValueError("SM90 bundle is missing a candidate or contains an unexpected candidate")
    variants = {}
    identity = None
    for name in policy()["configurations"]:
        entry = rows[name]
        if entry.get("directory") != name:
            raise ValueError("SM90 bundle member directory must match its configuration")
        member = directory / name
        if member.is_symlink() or not member.is_dir():
            raise ValueError("SM90 bundle member missing or symlinked")
        receipt_path = member / "build.json"
        if sha(receipt_path) != entry.get("receipt_sha256"):
            raise ValueError("SM90 build receipt changed after bundle assembly")
        receipt = json.loads(receipt_path.read_text())
        current = tuple(receipt.get(key) for key in BUILD_IDENTITY_FIELDS)
        if not all(current) or (identity is not None and current != identity):
            raise ValueError("SM90 candidates have mixed compiler/source/dependency/Torch identities")
        identity = current
        variants[name] = load_sm90(member)
    return AutoSm90Forward(variants)


@lru_cache(maxsize=None)
def _default_forward(override):
    if override is not None:
        if not override.strip():
            raise ValueError("ACTLIZE_LA_SM90_BUNDLE must not be empty")
        directory = Path(override)
        if not directory.is_absolute():
            raise ValueError("ACTLIZE_LA_SM90_BUNDLE requires an absolute directory")
    else:
        if not REGISTRATION_FILE.is_file():
            raise RuntimeError("SM90 auto bundle not installed; run bash tools/install_sm90.sh")
        registration = json.loads(REGISTRATION_FILE.read_text())
        if registration.get("schema") != 1:
            raise ValueError("invalid SM90 installation record")
        directory = Path(registration["directory"])
        if not directory.is_absolute() or sha(directory / "bundle.json") != registration.get("bundle_sha256"):
            raise ValueError("registered SM90 bundle changed; reinstall instead of guessing")
    return load_bundle(directory)


def default_forward():
    return _default_forward(os.getenv("ACTLIZE_LA_SM90_BUNDLE"))


def forward(q, k, v, g, beta, initial_state=None, output_final_state=True,
            *, mode="device", sm_count=None):
    return default_forward()(q, k, v, g, beta, initial_state, output_final_state,
                             mode=mode, sm_count=sm_count)
