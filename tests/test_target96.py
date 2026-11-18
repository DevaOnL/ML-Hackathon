"""Route96 transition policies preserve reachability and no-edit graph semantics."""
from dataclasses import replace
import unittest
import numpy as np
from mapmatch.target96 import RoutePolicy,RouteTargetMatcher,MemoEmissions,MemoIndex
from mapmatch.matcher import MatchConfig,MapMatcher
from mapmatch.candidates import Candidates
from mapmatch.data import Graph
from scripts.target96 import menu

class RouteTargetTests(unittest.TestCase):
    def test_policy_rejects_bad_numbers(self):
        for v in (-1,float('nan'),True):
            with self.assertRaises(ValueError):RoutePolicy(reverse_penalty=v)
    def test_menu_dense_retry_and_unique(self):
        values=menu();self.assertEqual(len(values),12);self.assertEqual(len({v['name'] for v in values}),12)
        for v in values:self.assertEqual(v['match_config']['candidate_retry_limit'],48);self.assertEqual(v['match_config']['observation_distance_m'],0)
    def test_memo_multiple_candidate_keys_recompute_without_overwrite(self):
        class Model:
            def __init__(self):self.calls=0
            def score_candidates_batch(self,g,t,i,c):self.calls+=1;return [np.full(len(x),1/len(x)) for x in c]
        m=Model();cache=MemoEmissions(m);g=object();track=np.zeros((1,3));a=[np.zeros(2)];b=[np.zeros(3)]
        p=cache.score_candidates_batch(g,track,[0],a);p[0][0]=0
        self.assertAlmostEqual(cache.score_candidates_batch(g,track,[0],a)[0][0],.5)
        cache.score_candidates_batch(g,track,[0],b);self.assertEqual(m.calls,2)
        with self.assertRaises(ValueError):cache.score_candidates_batch(object(),track,[0],a)
    def test_memo_rejects_nonfinite_scores(self):
        class Model:
            def score_candidates_batch(self,*args):return [np.array([float('nan')])]
        with self.assertRaises(ValueError):MemoEmissions(Model()).score_candidates_batch(object(),np.zeros((1,3)),[0],[np.zeros(1)])
if __name__=='__main__':unittest.main()
