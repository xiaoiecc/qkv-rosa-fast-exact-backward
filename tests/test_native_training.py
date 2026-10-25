"""The standalone C++ training core versus the previous Python calculation."""
from __future__ import annotations
import unittest
import numpy as np
import torch

from conv_rosa_transformer.rosa_layer import NBitQKVRosa
from conv_rosa_transformer.native_training import make_options,training_forward_numpy,training_backward_numpy
from tests.frozen_training_oracle import NBitQKVRosa as PythonOracle


class NativeTrainingTests(unittest.TestCase):
    def test_forward_cache_owns_inputs_and_options(self):
        for order in ("first", "second"):
            with self.subTest(order=order):
                rng=np.random.default_rng(382)
                logits=[rng.normal(size=(1,2,7,4)) for _ in range(3)]
                embeddings=[rng.normal(size=(2,4)) for _ in range(2)]
                options=make_options(credit_order=order,randomized=True,
                                     proposal_seed=17,repair_seed=23)
                _,cache,_=training_forward_numpy(*logits,*embeddings,options=options)
                gradient=rng.normal(size=logits[0].shape)
                before=training_backward_numpy(cache,gradient,trace=True)
                cached=(cache.q_logits,cache.k_logits,cache.v_logits,cache.emb0,cache.emb1)
                for original,saved in zip(logits+embeddings,cached):
                    self.assertFalse(np.shares_memory(original,saved))
                self.assertIsNot(options,cache.options)
                saved_options=bytes(cache.options)
                # Mutate every caller-owned input, including every option field.
                for original in logits+embeddings:
                    original.fill(np.nan)
                for field,_ in options._fields_:
                    setattr(options,field,0)
                self.assertEqual(bytes(cache.options),saved_options)
                after=training_backward_numpy(cache,gradient,trace=True)
                for a,b in zip(before[:5],after[:5]):
                    np.testing.assert_array_equal(a,b)
                self.assertEqual(before[-1].keys(),after[-1].keys())
                for key in before[-1]:
                    np.testing.assert_array_equal(before[-1][key],after[-1][key])

    def test_batched_native_matches_frozen_python_rule(self):
        for n,d in ((0,4),(1,4),(7,1),(9,4),(11,8),(6,64)):
          for mode in ("bernoulli","identity"):
           for second in (False,True):
            with self.subTest(n=n,d=d,mode=mode,second=second):
                config=dict(joint_pairs=1 if second else 0,trust_diagonal=.5 if second else 0,
                            joint_rows=32,joint_row_selection="active",credit_to_logit=mode,tau=.7)
                torch.manual_seed(491+n+d)
                native=NBitQKVRosa(2*d,d,**config).double()
                oracle=PythonOracle(2*d,d,**config).double();oracle.load_state_dict(native.state_dict())
                with torch.no_grad():
                    native.emb0.normal_();native.emb1.normal_();oracle.load_state_dict(native.state_dict())
                values=[torch.randn(2,n,2*d,dtype=torch.float64,requires_grad=True) for _ in range(3)]
                other=[z.detach().clone().requires_grad_() for z in values]
                gy=torch.randn(2,n,2*d,dtype=torch.float64)
                torch.manual_seed(919);actual=native._hard_rosa_from_qkv(*values)
                torch.manual_seed(919);expected=oracle._hard_rosa_from_qkv(*other)
                torch.testing.assert_close(actual,expected,atol=0,rtol=0)
                gradients=torch.autograd.grad(actual,values+[native.emb0,native.emb1],gy)
                reference=torch.autograd.grad(expected,other+[oracle.emb0,oracle.emb1],gy)
                for a,b in zip(gradients,reference):torch.testing.assert_close(a,b,atol=2e-11,rtol=2e-11)

    def test_native_exact_acceleration_preserves_all_arithmetic(self):
        rng=np.random.default_rng(817)
        for n,d in ((0,4),(9,4),(70,8),(15,64)):
          for order in ("first","second"):
            logits=[rng.normal(size=(2,2,n,d)) for _ in range(3)]
            e0,e1=rng.normal(size=(2,d)),rng.normal(size=(2,d));g=rng.normal(size=(2,2,n,d))
            outputs=[]
            for randomized in (False,True):
                y,cache,_=training_forward_numpy(*logits,e0,e1,credit_order=order,
                    randomized=randomized,proposal_seed=778,repair_seed=889,joint_rows=4)
                result=training_backward_numpy(cache,g,trace=True)
                outputs.append((y,result))
            self.assertTrue(np.array_equal(outputs[0][0],outputs[1][0]))
            for a,b in zip(outputs[0][1][:5],outputs[1][1][:5]):self.assertTrue(np.array_equal(a,b))
            for key in ("pair_codes","pair_costs"):
                self.assertTrue(np.array_equal(outputs[0][1][-1][key],outputs[1][1][-1][key],equal_nan=True))

    def test_native_replay_counts_and_full_model_contract(self):
        torch.manual_seed(25)
        layer=NBitQKVRosa(8,4).double();x=torch.randn(3,9,8,dtype=torch.float64,requires_grad=True)
        calls=[0,0]
        h1=layer.register_forward_hook(lambda *args:calls.__setitem__(0,calls[0]+1))
        h2=layer.register_full_backward_hook(lambda *args:calls.__setitem__(1,calls[1]+1))
        layer(x).square().mean().backward();h1.remove();h2.remove()
        self.assertEqual(calls,[1,1])
        work=layer.last_forward_work
        self.assertEqual(work["baseline_matching_calls"],6)
        self.assertEqual(work["backward"]["credit_engine_calls"],6)
        self.assertLessEqual(work["backward"]["joint_matching_calls"],6)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters()))

    def test_public_options(self):
        first=NBitQKVRosa(8,4,credit_order="first",randomized=True)
        self.assertEqual((first.joint_pairs,first.trust_diagonal),(0,0.))
        second=NBitQKVRosa(8,4,credit_order="second")
        self.assertEqual((second.joint_pairs,second.trust_diagonal),(1,.5))
        for kwargs in ({"credit_order":"invalid"},{"credit_order":"first","joint_pairs":1},
                       {"training_rule":"local_mixture"},{"stochastic_values":True}):
            with self.assertRaises(ValueError):NBitQKVRosa(8,4,**kwargs)
        with self.assertRaises(TypeError):NBitQKVRosa(8,4,randomized="yes")

    def test_full_repair_budget_does_not_consume_sampling_rng(self):
        layer=NBitQKVRosa(4,4,credit_order="first",repair_samples=4)
        x=torch.ones(1,5,4)
        state=torch.get_rng_state().clone()
        layer(x).sum().backward()
        self.assertTrue(torch.equal(state,torch.get_rng_state()))

    def test_float32_mapping_matches_previous_rule_with_roundoff_tolerance(self):
        torch.manual_seed(830)
        values=[torch.randn(2,17,8,requires_grad=True) for _ in range(3)]
        copies=[z.detach().clone().requires_grad_() for z in values]
        native=NBitQKVRosa(8,4)
        oracle=PythonOracle(8,4);oracle.load_state_dict(native.state_dict())
        gradient=torch.randn_like(values[0])
        torch.manual_seed(43);a=native._hard_rosa_from_qkv(*values)
        torch.manual_seed(43);b=oracle._hard_rosa_from_qkv(*copies)
        torch.testing.assert_close(a,b,atol=0,rtol=0)
        ga=torch.autograd.grad(a,values+[native.emb0,native.emb1],gradient)
        gb=torch.autograd.grad(b,copies+[oracle.emb0,oracle.emb1],gradient)
        for x,y in zip(ga,gb):torch.testing.assert_close(x,y,atol=2e-6,rtol=2e-6)

    def test_native_joint_corner_restores_real_barrier(self):
        q=np.array([[[[-.2,.2],[-.2,.2],[-.2,-.2]]]])
        k=np.array([[[[-.2,-.2],[.2,.2],[-.2,-.2]]]])
        v=np.array([[[[-.2,-.2],[-.2,-.2],[.2,.2]]]])
        e0=-np.ones((1,2));e1=np.ones((1,2));g=np.zeros_like(q);g[:,:,2]=-2
        # Fixed input/proposal seed: the C++ sampler selects a Q/Q pair.
        chosen=None
        for seed in range(30):
            _,cache,_=training_forward_numpy(q,k,v,e0,e1,proposal_seed=seed)
            result=training_backward_numpy(cache,g,trace=True)
            if tuple(result[-1]["pair_codes"][0,0])==(10,11):
                chosen=seed;break
        self.assertIsNotNone(chosen)
        p=1/(1+np.exp(.2))
        np.testing.assert_allclose(result[0][0,0,2],[-4*p*p*(1-p)]*2,rtol=1e-13,atol=1e-13)
        np.testing.assert_allclose(result[-1]["pair_costs"][0,0],[0,0,-4,-4],rtol=0,atol=1e-13)
        _,cache,_=training_forward_numpy(q,k,v,e0,e1,proposal_seed=chosen,trust_diagonal=1.)
        old=training_backward_numpy(cache,g,trace=True)
        np.testing.assert_array_equal(old[0][0,0,2],np.zeros(2))


if __name__=="__main__":unittest.main()
