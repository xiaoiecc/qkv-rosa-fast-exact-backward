"""Integration contracts for the explicit approximate backward."""
import unittest
import numpy as np
import torch
from conv_rosa_transformer.native_training import training_forward_numpy,training_backward_numpy,make_options
from conv_rosa_transformer.rosa_layer import NBitQKVRosa

class NativeApproximationTests(unittest.TestCase):
    def test_complete_residual_reconstructs_exact_with_same_joint_choices(self):
        rng=np.random.default_rng(891)
        for n,d in ((0,4),(1,4),(13,1),(17,4),(19,8),(11,64)):
          for order in ('first','second'):
            arrays=[rng.normal(size=(2,2,n,d)) for _ in range(3)]
            e0,e1=rng.normal(size=(2,2,d));g=rng.normal(size=arrays[0].shape)
            results=[];outputs=[]
            for mode in ('exact','approximate'):
                y,cache,_=training_forward_numpy(*arrays,e0,e1,credit_order=order,
                    proposal_seed=18,repair_seed=29,backward_mode=mode,approximation_rows=max(n,1))
                outputs.append(y);results.append(training_backward_numpy(cache,g,trace=True))
            np.testing.assert_array_equal(*outputs)
            for a,b in zip(results[0][:5],results[1][:5]):np.testing.assert_allclose(a,b,rtol=3e-12,atol=3e-12)
            # Embedding accumulators and V reuse the complete native loop.
            for i in (2,3,4):np.testing.assert_array_equal(results[0][i],results[1][i])
            np.testing.assert_array_equal(results[0][-1]['pair_codes'],results[1][-1]['pair_codes'])
            np.testing.assert_allclose(results[0][-1]['pair_costs'],results[1][-1]['pair_costs'],rtol=3e-12,atol=3e-12)
            self.assertTrue(results[1][-1]['approximation_complete_residual'])

    def test_partial_sampling_keeps_v_embeddings_and_selected_joint_exact(self):
        rng=np.random.default_rng(129)
        n,d=71,3
        arrays=[rng.normal(size=(1,2,n,d)) for _ in range(3)]
        # Long repeating keys/queries exercise nonzero row residuals.
        arrays[0][:]=-.2
        arrays[1][:]=arrays[0]
        arrays[1][:,:,0,0]=.2
        e0,e1=rng.normal(size=(2,2,d));g=rng.normal(size=arrays[0].shape)
        results=[]
        for mode in ('exact','approximate'):
            _,cache,_=training_forward_numpy(*arrays,e0,e1,backward_mode=mode,
                approximation_rows=4,proposal_seed=91,repair_seed=67)
            results.append(training_backward_numpy(cache,g,trace=True))
        for i in (2,3,4):np.testing.assert_array_equal(results[0][i],results[1][i])
        np.testing.assert_array_equal(results[0][-1]['pair_codes'],results[1][-1]['pair_codes'])
        np.testing.assert_allclose(results[0][-1]['pair_costs'],results[1][-1]['pair_costs'],rtol=2e-12,atol=2e-12)
        self.assertFalse(results[1][-1]['approximation_complete_residual'])
        self.assertEqual(results[1][-1]['approx_selected_rows'],8)

    def test_normal_autograd_rng_and_saved_configuration(self):
        for order in ('first','second'):
            torch.manual_seed(817)
            layer=NBitQKVRosa(8,4,credit_order=order,backward_mode='approximate',approximation_rows=4).double()
            x=torch.randn(2,37,8,dtype=torch.float64,requires_grad=True)
            state=torch.get_rng_state().clone();y=layer(x);after=torch.get_rng_state().clone()
            self.assertFalse(torch.equal(state,after))
            layer.backward_mode='exact';layer.approximation_rows=1
            y.square().mean().backward()
            self.assertEqual(layer.last_forward_work['backward']['backward_mode'],'approximate')
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters()))
            self.assertTrue(torch.equal(after,torch.get_rng_state()))
        first=NBitQKVRosa(4,4,credit_order='first');state=torch.get_rng_state().clone()
        first(torch.ones(1,8,4)).sum().backward()
        self.assertTrue(torch.equal(state,torch.get_rng_state()))

    def test_validation_and_defaults(self):
        self.assertTrue(make_options().randomized)
        self.assertTrue(NBitQKVRosa(8,4).randomized)
        for kw in ({'backward_mode':'bad'},{'approximation_rows':0},
                   {'backward_mode':'approximate','repair_samples':2}):
            with self.assertRaises(ValueError):NBitQKVRosa(8,4,**kw)

if __name__=='__main__':unittest.main()
