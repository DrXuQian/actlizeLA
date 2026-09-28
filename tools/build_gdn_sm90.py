#!/usr/bin/env python3
"""Build the complete independent SM90 graph, without importing legacy AIU.

ppu17/source-check = real CUDA SM90a assembly against the PPU CUTLASS fork.
It is NOT native PPU code or device admission. A native PPU target probe must
pass with the very same flags before compiling the implementation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import sysconfig

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "csrc/backends/sm90"
CONFIGURATIONS = ("control", "value64", "value64-local-inverse", "value128-paired")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def flags(target, mode, configuration="control"):
    if configuration not in CONFIGURATIONS:
        raise ValueError(f"unknown configuration: {configuration}")
    result = ["-std=c++17", "-O3", "--expt-relaxed-constexpr", "--extended-lambda",
              "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo", "-Xcompiler=-fPIC"]
    if target == "ppu17":
        result += ["-DGDN_SM90_PPU17=1", "-DACOMPUTE_VERSION=10700"]
        if mode == "source-check":
            result += ["-DGDN_SM90_SOURCE_CHECK=1"]
    result += [f"-DGDN_SM90_CONFIGURATION={CONFIGURATIONS.index(configuration)}"]
    if configuration != "control":
        result += ["-DNDEBUG"]
    return result


def dependency(target, supplied):
    root = Path(supplied or ROOT / "third_party/cutlass").resolve()
    if target == "ppu17":
        if not supplied:
            raise ValueError("ppu17 requires an explicit PPU_CUTLASS_ROOT; never borrow actlize")
        version = (root / "include/cutlass/version.h").read_text()
        parts = tuple(int(re.search(rf"#define CUTLASS_{p}\s+(\d+)",version)[1])
                      for p in ("MAJOR","MINOR","PATCH"))
        if parts != (3,6,0) or not (root / "include/ppu/ppu_include_10700.hpp").is_file():
            raise ValueError("ppu17 requires PPU CUTLASS 3.6.0 with the 10700 backend")
    if not (root / "include/cute/tensor.hpp").is_file():
        raise ValueError(f"missing CUTLASS dependency: {root}")
    return root


def dependency_identity(root):
    """Bind actual headers, including untracked files and archive snapshots.

    Git searches parent directories. Its HEAD is the dependency's own revision
    only when the discovered top-level is exactly this dependency directory.
    """
    root = Path(root).resolve()
    files = {}
    for path in sorted((root / "include").rglob("*")):
        if path.is_symlink() and path.is_dir():
            raise ValueError(f"dependency include directory symlink requires a canonical snapshot: {path}")
        if path.is_file():
            files[path.relative_to(root).as_posix()] = sha(path)
    if not files:
        raise ValueError(f"empty dependency include tree: {root}")
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    probe = subprocess.run(["git", "-C", str(root), "rev-parse", "--show-toplevel"],
                           capture_output=True, text=True)
    revision = None
    containing = None
    if probe.returncode == 0:
        top = Path(probe.stdout.strip()).resolve()
        head = subprocess.check_output(["git", "-C", str(top), "rev-parse", "HEAD"],
                                       text=True).strip()
        if top == root:
            revision = head
        else:
            containing = dict(root=str(top), revision=head,
                              subtree=root.relative_to(top).as_posix())
    return dict(dependency_revision=revision,
                dependency_containing_repository=containing,
                dependency_tree_sha256=hashlib.sha256(encoded).hexdigest(),
                dependency_headers_sha256=files)


def admit_reuse(prior,current,obj):
    keys=("target","mode","configuration","compiler","compiler_sha256","dependency","flags","include",
          "source_sha256","dependency_version_sha256","dependency_tree_sha256")
    if not prior.get("dependency_tree_sha256") or not current.get("dependency_tree_sha256"):
        raise ValueError("device reuse dependency identity changed or lacks a complete header hash; rebuild")
    if any(prior.get(k)!=current.get(k) for k in keys):
        raise ValueError("device reuse identity/source/dependency changed; rebuild in a new directory")
    if not obj.is_file() or prior.get("device_object_sha256")!=sha(obj):
        raise ValueError("device reuse object missing or hash mismatch")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target",choices=("cuda_sm90","ppu17"),required=True)
    p.add_argument("--mode",choices=("native","source-check"),default="native")
    p.add_argument("--configuration", choices=CONFIGURATIONS, default="control")
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--compiler",type=Path)
    p.add_argument("--cutlass-root",default=os.getenv("PPU_CUTLASS_ROOT"))
    p.add_argument("--device-only",action="store_true")
    p.add_argument("--reuse-device",action="store_true",help="reuse only an exactly hash-bound device object; recheck target/codegen/link/import")
    args=p.parse_args()
    out=args.out.resolve(); out.mkdir(parents=True,exist_ok=True)
    scratch=out/"scratch"; scratch.mkdir(exist_ok=True)
    env=os.environ|{"TMPDIR":str(scratch)}
    dep=dependency(args.target,args.cutlass_root)
    if args.compiler:
        compiler=args.compiler.resolve()
    elif args.target=="ppu17" and args.mode=="native":
        sdk=os.getenv("PPU_SDK_ROOT",os.getenv("PPU_SDK"))
        if not sdk: raise ValueError("native ppu17 requires PPU_SDK_ROOT or --compiler")
        compiler=Path(sdk).resolve()/"CUDA_SDK/bin/nvcc"
    else:
        compiler=Path(os.getenv("CUDA_HOME","/usr/local/cuda"))/"bin/nvcc"
    if not compiler.is_file(): raise ValueError(f"compiler missing: {compiler}")
    # Keep wrapper path: SDK nvcc wrappers may locate their runtime relative to it.
    include=[f"-I{SOURCE}",f"-I{SOURCE/'cula'}",f"-I{dep/'include'}"]
    options=flags(args.target,args.mode,args.configuration)
    identity=dict(target=args.target,mode=args.mode,configuration=args.configuration,compiler=str(compiler),compiler_sha256=sha(compiler),
                  dependency=str(dep),flags=options,include=include,device_admission="NOT_RUN")
    identity["dependency_version_sha256"]=sha(dep/"include/cutlass/version.h")
    identity["repository_revision"]=subprocess.check_output(
        ["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip()
    identity.update(dependency_identity(dep))
    identity["source_sha256"]={str(x.relative_to(ROOT)):sha(x) for x in sorted(SOURCE.rglob("*")) if x.is_file()}
    identity_file=out/"build.json"
    prior={}
    if identity_file.exists():
        prior=json.loads(identity_file.read_text())
        if any(prior.get(k)!=identity.get(k) for k in ("target","mode","compiler","compiler_sha256","dependency","flags")):
            raise ValueError("build identity changed: use a new output directory")
    if args.reuse_device: admit_reuse(prior,identity,out/"launch.o")
    identity_file.write_text(json.dumps(identity,indent=2)+"\n")
    def run(label,cmd):
        print(f"[GDN SM90 build] {label}: {shlex.join(map(str,cmd))}",flush=True)
        with (out/f"{label}.log").open("w") as log:
            log.write(shlex.join(map(str,cmd))+"\n"); log.flush()
            result=subprocess.run(list(map(str,cmd)),stdout=log,stderr=subprocess.STDOUT,env=env)
        if result.returncode:
            raise RuntimeError(f"{label} failed rc={result.returncode}; see {out/f'{label}.log'}")
    run("target",[compiler,*options,*include,"-c",ROOT/"dev/backends/sm90_target_probe.cu","-o",out/"target.o"])
    if not args.reuse_device:
        run("device",[compiler,*options,*include,"-Xptxas=-v","-c",SOURCE/"launch.cu","-o",out/"launch.o"])
    else:
        print("[GDN SM90 build] device reuse=EXACT_SOURCE_FLAGS_DEPENDENCY_OBJECT_HASH",flush=True)
    identity["device_object_sha256"]=sha(out/"launch.o")
    if args.target=="cuda_sm90" or args.mode=="source-check":
        run("codegen",[sys.executable,ROOT/"dev/backends/check_sm90_binary.py",out/"launch.o",
            "--cuobjdump",compiler.parent/"cuobjdump","--out",out/"codegen"])
    if not args.device_only:
        import torch
        from torch.utils.cpp_extension import include_paths,library_paths
        cuda=compiler.parent.parent
        cuda_include=cuda/"include"
        cuda_lib=cuda/"lib64"
        target_name=args.target+("-source-check" if args.target=="ppu17" and args.mode=="source-check" else "")
        macros=[f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
                "-DTORCH_EXTENSION_NAME=_gdn_fused_sm90",f'-DGDN_SM90_TARGET_NAME="{target_name}"',
                *(x for x in options if x.startswith("-D"))]
        cxx=os.getenv("CXX","c++")
        run("host",[cxx,"-std=c++17","-O3","-fPIC",*macros,f"-I{SOURCE}",
            f"-I{sysconfig.get_path('include')}",f"-I{cuda_include}",
            *(f"-I{x}" for x in include_paths()),"-c",SOURCE/"bindings.cpp","-o",out/"bindings.o"])
        library_dirs=[*library_paths(),str(cuda_lib)]
        extension=out/("_gdn_fused_sm90"+sysconfig.get_config_var("EXT_SUFFIX"))
        run("link",[cxx,"-shared",out/"bindings.o",out/"launch.o",
            *(f"-L{x}" for x in library_dirs),*(f"-Wl,-rpath,{x}" for x in library_dirs),
            "-ltorch_python","-ltorch","-ltorch_cpu","-ltorch_cuda","-lc10","-lc10_cuda","-lcudart",
            "-o",extension])
        identity.update(extension=str(extension),extension_sha256=sha(extension),torch=torch.__version__)
        run("import",[sys.executable,"-c",
            "import importlib.util,torch; "
            f"s=importlib.util.spec_from_file_location('_gdn_fused_sm90',{str(extension)!r}); "
            "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); "
            f"assert m.target=={target_name!r}; "
            f"assert m.configuration=={args.configuration!r}; "
            "assert m.math_contract=='cula-scalar-gdn-fused-bf16-v1'; "
            "print('IMPORT/PASS; no device queried/launched')"])
        print(f"[GDN SM90 build] extension={extension}",flush=True)
    after={str(x.relative_to(ROOT)):sha(x) for x in sorted(SOURCE.rglob("*")) if x.is_file()}
    if after!=identity["source_sha256"]: raise RuntimeError("source changed during build; do not use this binary")
    if dependency_identity(dep)["dependency_tree_sha256"] != identity["dependency_tree_sha256"]:
        raise RuntimeError("dependency headers changed during build; do not use this binary")
    identity["complete"]=True
    identity_file.write_text(json.dumps(identity,indent=2)+"\n")
    print(f"[GDN SM90 build] PASS scope=COMPILE_LINK_ONLY target={args.target} mode={args.mode} device=NOT_RUN")


if __name__=="__main__":
    try: main()
    except (ValueError,RuntimeError,OSError) as error:
        print(f"[GDN SM90 build] FAIL: {error}",file=sys.stderr)
        sys.exit(1)
