"""Single-TF32 numerical screens and fail-closed source/runner controls."""
import importlib
import copy
from pathlib import Path
import subprocess
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for directory in ("tools", "dev/ppu", "benchmarks"):
    sys.path.insert(0, str(ROOT / directory))
check = importlib.import_module("check_solve_tf32x1")
contract = importlib.import_module("gdn_kda_contract")


def tf32(x):
    # Match actlize tfloat32_t: add a half ULP then discard 13 mantissa bits.
    # This is not a claim that all TF32 backends use nearest-even conversion.
    x = x.float().contiguous()
    bits = x.view(torch.int32)
    return ((bits + 0x1000) & -8192).view(torch.float32)


def inverse(lower, plant=False):
    n = lower.shape[0]
    assert n == 64
    result = torch.zeros_like(lower)
    for begin in range(0, n, 16):
        block = slice(begin, begin + 16)
        eye = torch.eye(16)
        result[block, block] = torch.linalg.solve_triangular(
            eye + lower[block, block], eye, upper=False, unitriangular=True)
    product = lambda a,b: tf32(a) @ tf32(b)
    for gap in range(1, 4):
        for bc in range(4 - gap):
            br = bc + gap
            row, col = slice(br*16,(br+1)*16), slice(bc*16,(bc+1)*16)
            acc = torch.zeros(16,16)
            for k in range(bc, br):
                middle = slice(k*16,(k+1)*16)
                acc += product(lower[row,middle], result[middle,col])
            result[row,col] = (1 if plant else -1) * product(result[row,row], acc)
    return result


@pytest.mark.parametrize("valid", [1, 15, 16, 17, 31, 32, 33, 63, 64])
def test_blocked_inverse_partial_extent_and_wrong_sign(valid):
    gen = torch.Generator().manual_seed(146)
    key = torch.randn(64,128,generator=gen) * .05
    beta = torch.rand(64,generator=gen)
    g = -torch.rand(64,generator=gen) * .2
    prefix = g.cumsum(0)
    lower = torch.tril((key @ key.T) * beta[:,None] * (prefix[:,None]-prefix[None,:]).exp(), -1)
    lower[valid:] = 0
    lower[:,valid:] = 0
    eye = torch.eye(64)
    got = inverse(lower)
    reference = torch.linalg.solve_triangular((eye+lower).double(),eye.double(),upper=False,unitriangular=True).float()
    assert (got-reference).abs().max() < 1e-4
    assert ((eye+lower) @ got-eye).abs().max() < 1e-4
    if valid > 16:
        assert ((eye+lower) @ inverse(lower, plant=True)-eye).abs().max() > 1e-4


def residual_screen(inputs, initial):
    # FP32 CPU matrix products emulate rounding boundaries, NOT native MMA
    # instruction accumulation or physical layouts. Device gate remains required.
    q,k,v,g,beta = inputs
    _,length,hk,_ = q.shape
    hv = v.shape[2]
    outputs, finals = [], []
    bf = lambda x: x.bfloat16().float()
    for head in range(hv):
        h = initial[0,head].clone()
        blocks = []
        for start in range(0,length,64):
            sl = slice(start,start+64)
            qq,kk,vv = q[0,sl,head//(hv//hk)].float(),k[0,sl,head//(hv//hk)].float(),v[0,sl,head].float()
            prefix,bb = g[0,sl,head].float().cumsum(0),beta[0,sl,head].float()
            decay = (prefix[:,None]-prefix[None,:]).exp()
            lower = torch.tril((kk @ kk.T) * bb[:,None] * decay,-1)
            p = bf(inverse(lower))
            snapshot = bf(h)
            residual = bf(bb[:,None] * (vv - prefix.exp()[:,None] * (kk @ snapshot)))
            value = p @ residual
            attn = bf(torch.tril((qq @ kk.T) * decay))
            blocks.append(((prefix.exp()[:,None]*(qq @ snapshot) + attn @ bf(value))*(128**-.5)).bfloat16())
            h = prefix[-1].exp()*h + kk.T @ bf(value*(prefix[-1]-prefix).exp()[:,None])
        outputs.append(torch.cat(blocks))
        finals.append(h)
    return torch.stack(outputs,1).unsqueeze(0),torch.stack(finals).unsqueeze(0)


@pytest.mark.parametrize("gate", [-.1, -1.])
@pytest.mark.parametrize("length,hv", [(128,3), (2048,1), (8192,1)])
def test_single_tf32_full_recurrence_preserves_original_numeric_limit(gate,length,hv):
    from test_ppu_gdn_backend import fixture, assert_pair
    from test_ppu_wy_backend import oracle
    inputs = fixture(1,length,1,hv,gate)
    state = torch.randn(1,hv,128,128,generator=torch.Generator().manual_seed(88))*.02
    got, want = residual_screen(inputs,state),oracle(inputs,state)
    assert_pair(got,want)
    for which in (0,1):
        broken = list(got)
        broken[which] = torch.zeros_like(broken[which])
        with pytest.raises(AssertionError):
            assert_pair(broken,want)


def test_source_seam_and_wrong_indices_are_rejected():
    texts = check.source_texts()
    check.check_source(*texts)
    for old,new in (("single_tf32_product(acc,","tf32_product(acc,"),
                    ("b[(rc.col + k) * ldb + rc.row]","b[rc.row * ldb + rc.col + k]"),
                    ("= -merged[s]","= merged[s]"),
                    ("solve_tf32x1::launch_inverse(p, inverse_ws, stream)","solve_static::launch_inverse_default(p, inverse_ws, stream)"),
                    ("sequence % Chunk ||","")):
        assert old in texts[1]
        changed = [texts[0],texts[1].replace(old,new,1),texts[2]]
        with pytest.raises(AssertionError):
            check.check_source(*changed)


def test_default_sources_stay_identical_to_round_one():
    for name in ("csrc/gdn_chunk/gdn_wy_solve_static_ppu.cu", "include/gdn_qsa/ppu/wy_mma.cuh",
                 "csrc/backends/ppu10/launch.cu", "csrc/gdn_chunk/gdn_wy_residual_full_chunk_ppu.cu"):
        assert (ROOT/name).read_bytes() == subprocess.check_output(["git","show","3d25e7e:"+name],cwd=ROOT)


def test_tf32_inventory_cannot_accept_old_solve_or_wu_stacking():
    names = contract.EXPERIMENTS['solve-tf32x1']['kernels']
    rows = [dict(kernel=name,process='1',us=1.) for name in names]
    assert contract.inventory('solve-tf32x1',rows)['status']=='ADMITTED'
    rows[1]['kernel']='gdn_wy_split_solve_static'
    assert contract.inventory('solve-tf32x1',rows)['status']=='UNADJUDICATED'
    rows[1]['kernel']=names[1]
    rows.append(dict(kernel='gdn_wy_parallel_wu_materialize',process='1',us=100.))
    bad=contract.inventory('solve-tf32x1',rows)
    assert bad['status']=='UNADJUDICATED' and bad['total_us']==104.


def test_new_experiment_verdict_and_identity_are_not_aliased_to_round_one():
    selected = contract.roles('solve-tf32x1')
    samples = {role:[dict(status='ADMITTED',total_us=t+i/10) for i in range(3)]
               for role,t in zip(selected,(12.,10.,11.))}
    result = contract.summarize(samples,'solve-tf32x1')
    assert result['comparisons']['default_vs_solve-tf32x1']['verdict']=='solve-tf32x1_WINS'
    assert result['target_candidate_vs_fla']=='TARGET_NOT_MET'
    with pytest.raises(ValueError):
        contract.summarize(samples)  # Wrong experiment must not reuse a verdict.
    admission = dict(status='PASS',numeric_repeats=8,unchanged_public_default_bits='PASS',
                     candidate_kind='solve-tf32x1',device={},shape={},gate=-.1,libraries={},
                     operands_sha='i',input_admission_sha='p',canonical_input_sha='c',
                     arms={r:dict(input_sha='i',output_sha='o',route={'role':r}) for r in selected})
    subject = dict(admission,phase='subject',role='solve-tf32x1',public_calls=1,warmup=0,verification='CPU',
                   **admission['arms']['solve-tf32x1'])
    contract.validate_receipt(admission,subject,'solve-tf32x1')
    changed=copy.deepcopy(admission)
    changed['candidate_kind']='parallel-wu'
    with pytest.raises(ValueError):
        contract.validate_receipt(changed,subject,'solve-tf32x1')


@pytest.mark.parametrize("target", ['ppu15','cuda_sm120'])
def test_single_tf32_is_explicit_ppu10_opt_in(tmp_path,target):
    result = subprocess.run(['cmake','-S',ROOT/'csrc/serving','-B',tmp_path/'build',
                             '-DALA_TARGET='+target,'-DALA_PPU10_SOLVE_TF32X1_EXPERIMENT=ON'],
                             text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    assert result.returncode and 'Single-TF32 experiment requires explicit ppu10' in result.stdout
