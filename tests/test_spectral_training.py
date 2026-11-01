"""Public training contracts for shared long-corridor numerical contraction."""
import unittest
import numpy as np
from conv_rosa_transformer.native_training import training_forward_numpy,training_backward_numpy


class SpectralTrainingTests(unittest.TestCase):
    def case(self,n,d,seed,payload_period=None,sparse=False):
        rng=np.random.default_rng(seed)
        q=np.full((1,1,n,d),-.3);k=q.copy();k[:,:,0,-1]=.3
        v=(2*rng.integers(0,2,q.shape)-1)*.3
        if payload_period is not None:
            v[:]=np.resize(v[:,:,:payload_period],v.shape)
        e0,e1=rng.normal(size=(2,1,d));g=rng.normal(size=q.shape)
        if sparse:g[:,:,:-1]=0
        return [q,k,v,e0,e1],g

    def compare(self,args,g,order):
        result=[]
        for mode in ('exact','approximate'):
            _,cache,_=training_forward_numpy(*args,credit_order=order,backward_mode=mode,
                proposal_seed=341,repair_seed=719,approximation_rows=32)
            result.append(training_backward_numpy(cache,g,trace=True))
        for i in (2,3,4):self.assertEqual(result[0][i].tobytes(),result[1][i].tobytes())
        np.testing.assert_array_equal(result[0][-1]['pair_codes'],result[1][-1]['pair_codes'])
        np.testing.assert_allclose(result[0][-1]['pair_costs'],result[1][-1]['pair_costs'],atol=1e-10,rtol=1e-12)
        return result

    def test_long_groups_keep_complete_routes_and_small_roundoff(self):
        args,g=self.case(2048,8,791)
        for order in ('first','second'):
            exact,approx=self.compare(args,g,order)
            self.assertGreater(approx[-1]['approx_spectral_groups'],0)
            self.assertEqual(approx[-1]['approx_selected_rows'],0)
            for i in (0,1):np.testing.assert_allclose(exact[i],approx[i],rtol=3e-11,atol=3e-11)

    def test_zero_score_recovery_does_not_invent_a_gradient(self):
        args,g=self.case(2048,64,341,payload_period=2)
        for order in ('first','second'):
            exact,approx=self.compare(args,g,order)
            for i in (0,1):
                np.testing.assert_allclose(exact[i],approx[i],rtol=3e-11,atol=3e-11)
                self.assertTrue(np.all(approx[i][exact[i]==0]==0))

    def test_sparse_adjoint_uses_complete_small_support(self):
        args,g=self.case(1024,8,431,sparse=True)
        exact,approx=self.compare(args,g,'second')
        self.assertEqual(approx[-1]['approx_spectral_groups'],0)
        self.assertTrue(approx[-1]['approximation_complete_residual'])
        for a,b in zip(exact[:5],approx[:5]):np.testing.assert_allclose(a,b,rtol=3e-12,atol=3e-12)

    def test_explicit_channel_budget_preserves_complete_corners_and_rng(self):
        args,g=self.case(2048,64,716)
        outputs=[]
        for budget in (0,8,8):
            _,cache,_=training_forward_numpy(*args,backward_mode='approximate',
                approximation_channels=budget,proposal_seed=341,repair_seed=719)
            outputs.append(training_backward_numpy(cache,g,trace=True))
        full,sampled,repeated=outputs
        self.assertLess(sampled[-1]['approx_spectral_transforms'],full[-1]['approx_spectral_transforms'])
        self.assertEqual(sampled[-1]['approximation_channels'],8)
        for a,b in zip(sampled[:5],repeated[:5]):self.assertEqual(a.tobytes(),b.tobytes())
        for i in (2,3,4):self.assertEqual(sampled[i].tobytes(),full[i].tobytes())
        np.testing.assert_array_equal(sampled[-1]['pair_codes'],full[-1]['pair_codes'])
        np.testing.assert_allclose(sampled[-1]['pair_costs'],full[-1]['pair_costs'],rtol=1e-12,atol=1e-10)

    def test_channel_budget_and_external_cache_bounds(self):
        args,g=self.case(7,4,317)
        for invalid in (-1,65):
            with self.assertRaises(ValueError):
                training_forward_numpy(*args,backward_mode='approximate',approximation_channels=invalid)
        _,cache,_=training_forward_numpy(*args,backward_mode='approximate')
        cache.lengths[0,0,1]=2
        with self.assertRaises(RuntimeError):training_backward_numpy(cache,g)


if __name__=='__main__':unittest.main()
