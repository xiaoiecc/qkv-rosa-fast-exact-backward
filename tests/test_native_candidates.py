"""Native candidate parity against the retained Python-only test oracle."""
import unittest
import numpy as np
from conv_rosa_transformer import rosa_core
from conv_rosa_transformer.causal_candidates import propose_pairs
from tests.causal_candidates_oracle import propose_pairs as python_oracle


class NativeCandidateParityTests(unittest.TestCase):
    def test_full_capacity_pool_priority_and_main_work(self):
        rng=np.random.default_rng(2711021)
        for d in (1,2,4,8,64):
            for n in (1,2,9,25):
                for seed in (0,1,991):
                    q=rng.bit_generator.random_raw(n);k=rng.bit_generator.random_raw(n)
                    if d<64:
                        q&=np.uint64((1<<d)-1);k&=np.uint64((1<<d)-1)
                    options=dict(max_pairs=8,seed=seed,bucket_cap=n,occurrence_cap=n,
                                 scan=n,candidate_cap=10000,candidate_trace=True)
                    old,previous=python_oracle(q,k,d,**options)
                    new,current=propose_pairs(q,k,d,**options)
                    self.assertEqual(old,new)
                    self.assertEqual(previous['candidate_pairs'],current['candidate_pairs'])
                    for name in ('examined_alignments','symbol_comparisons','candidate_events',
                                 'retained_candidates','selected_pairs'):
                        self.assertEqual(previous[name],current[name])

    def test_public_numpy_boundary_and_empty_rows(self):
        q,k=[0,3],[0,0]
        args=[q,k,2,[1],[1,1],[1,1],2,7,8,2,8,16]
        pairs,stats=rosa_core.pair_candidates_numpy(*args,candidate_trace=True)
        self.assertEqual(len(pairs),2)
        self.assertEqual(stats['selected_pairs'],2)
        empty=args.copy();empty[3]=[]
        self.assertEqual(len(rosa_core.pair_candidates_numpy(*empty)[0]),0)
        for index,bad in ((2,0),(3,[2]),(3,[.5]),(4,[1]),(5,[2,0]),
                          (6,-1),(7,-1),(7,2**64),(8,0),(9,0),(10,0),(11,1)):
            altered=args.copy();altered[index]=bad
            with self.subTest(index=index,bad=bad):
                with self.assertRaises(ValueError):
                    rosa_core.pair_candidates_numpy(*altered)


if __name__=='__main__':unittest.main()
