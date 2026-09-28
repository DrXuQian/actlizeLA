"""FP64 derivation admission, NOT an emulation of device rounding or races.

Compare cuLA's inverse/residual chunk formula with an independently ordered
token recurrence. Negative controls exercise the new GDN integration seams.
"""
import unittest
import torch

torch.set_num_threads(1)


def recurrent(q,k,v,g,beta,state):
    h=state.clone(); result=[]
    for t in range(q.shape[0]):
        h=h*g[t].exp()
        residual=v[t]-k[t]@h
        h=h+k[t,:,None]*(beta[t]*residual)[None,:]
        result.append(q[t]@h/q.shape[-1]**.5)
    return torch.stack(result),h


def fused_formula(q,k,v,g,beta,state,plant=None):
    h=state.T.clone() if plant=="state-transpose" else state.clone()
    out=[]; carry=0.
    for start in range(0,len(q),64):
        qc,kc,vc,bc=q[start:start+64],k[start:start+64],v[start:start+64],beta[start:start+64]
        prefix=g[start:start+64].cumsum(0)
        if plant=="cross-chunk-prefix": prefix=prefix+carry
        carry=prefix[-1]
        log2=prefix/torch.log(torch.tensor(2.,dtype=torch.float64))
        if plant=="wrong-log-base": log2=prefix
        decay=(log2[:,None]-log2[None,:]).exp2()
        lower=torch.tril((kc@kc.T)*decay*bc[:,None],diagonal=-1)
        inv=torch.linalg.solve_triangular(torch.eye(len(qc),dtype=q.dtype)+lower,
                                         torch.eye(len(qc),dtype=q.dtype),upper=False)
        # cuLA's inverse is right-scaled by beta before consuming residual.
        if plant=="wrong-beta-axis": inv=bc[:,None]*inv
        else: inv=inv*bc[None,:]
        delta=inv@(vc-(kc*log2.exp2()[:,None])@h)
        o=(qc*log2.exp2()[:,None])@h+torch.tril((qc@kc.T)*decay)@delta
        out.append(o/q.shape[-1]**.5)
        h=h*log2[-1].exp2()+(kc*(log2[-1]-log2).exp2()[:,None]).T@delta
    return torch.cat(out),h


def data(length,gate,seed=33):
    gen=torch.Generator().manual_seed(seed)
    q,k,v=[torch.randn(length,128,generator=gen,dtype=torch.float64)*.05 for _ in range(3)]
    g=torch.rand(length,generator=gen,dtype=torch.float64)*gate
    beta=torch.rand(length,generator=gen,dtype=torch.float64)*.7+.1
    state=torch.randn(128,128,generator=gen,dtype=torch.float64)*.03
    return q,k,v,g,beta,state


class Algebra(unittest.TestCase):
    def test_scalar_kda_is_gdn_with_tail_and_initial_state(self):
        count=0
        for length in (1,16,31,32,33,63,64,65,127,128,129,257):
            for gate in (0.,-.1,-1.):
                for initial in (False,True):
                    inputs=list(data(length,gate))
                    if not initial: inputs[-1].zero_()
                    for got,want in zip(fused_formula(*inputs),recurrent(*inputs)):
                        torch.testing.assert_close(got,want,rtol=1e-11,atol=1e-12)
                    count+=1
        print(f"[SM90 algebra] cases={count} scalar-KDA=GDN FP64/PASS; not device admission")

    def test_new_seam_negatives(self):
        inputs=data(129,-.1)
        want=recurrent(*inputs)
        for plant in ("state-transpose","cross-chunk-prefix","wrong-log-base","wrong-beta-axis"):
            got=fused_formula(*inputs,plant=plant)
            with self.subTest(plant=plant), self.assertRaises(AssertionError):
                for a,b in zip(got,want):
                    torch.testing.assert_close(a,b,rtol=1e-11,atol=1e-12)
            print(f"[SM90 negative] {plant} EXPECTED-RED/PASS")


if __name__=="__main__": unittest.main()
