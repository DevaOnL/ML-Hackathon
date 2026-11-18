"""Exact Dijkstra hop prior, continuity, zero-policy parity and immutable scoring."""
from dataclasses import replace
import unittest
import numpy as np
from mapmatch.data import Graph,validate_route
from mapmatch.candidates import Candidates
from mapmatch.matcher import MapMatcher,MatchConfig
from mapmatch.target96_arc import HopRouter,ArcPriorMatcher,ArcPolicy

class ArcPriorTests(unittest.TestCase):
    def graph(self):
        return Graph(np.array([[0,0],[.001,0],[.002,0],[.003,0],[.004,0]],float),np.array([[0,1],[1,2],[2,3],[3,4],[2,1],[3,2]],int))
    def test_hops_match_exact_paths_horizon_and_eviction(self):
        g=self.graph();r=HopRouter(g,cache_size=1)
        for horizon in (150,500,1000,100):
            for source in (0,1,2,0):
                expected=[]
                for target in range(len(g.nodes)):
                    d,p=r.shortest_path(source,target,horizon);expected.append(len(p) if np.isfinite(d) else np.inf)
                np.testing.assert_array_equal(r.hop_counts(source,list(range(len(g.nodes))),horizon),expected)
    def test_arc_prior_counts_same_direct_zero_and_bridge(self):
        g=self.graph();cfg=MatchConfig(sigma_m=10,beta_m=2)
        a=Candidates(np.array([0,1]),np.array([.5,.9]),np.zeros(2));b=Candidates(np.array([1,3]),np.array([.95,.5]),np.zeros(2))
        m=ArcPriorMatcher(g,cfg,None,ArcPolicy(appended_arc_penalty=.5));counts=m._appended_counts(a,b,1000)
        self.assertEqual(counts[0,0],1);self.assertEqual(counts[0,1],3);self.assertEqual(counts[1,0],0);self.assertEqual(counts[1,1],2)
        base=MapMatcher(g,cfg)._transitions(a,b,10,1000);actual=m._transitions(a,b,10,1000)
        np.testing.assert_allclose(actual,base+.5*counts)
    def test_zero_policy_route_parity(self):
        g=self.graph();track=np.array([[.0001,0,0],[.0009,0,1],[.0015,0,2],[.0039,0,3]],float);cfg=MatchConfig()
        a=MapMatcher(g,cfg).match(track);b=ArcPriorMatcher(g,cfg,None).match(track)
        np.testing.assert_array_equal(a.route,b.route);np.testing.assert_array_equal(a.matched_edges,b.matched_edges)
        self.assertTrue(validate_route(g,b.route)['valid'])
    def test_positive_prior_preserves_infinity(self):
        g=self.graph();a=Candidates(np.array([3]),np.array([.9]),np.zeros(1));b=Candidates(np.array([0]),np.array([.1]),np.zeros(1));m=ArcPriorMatcher(g,MatchConfig(),None,ArcPolicy(appended_arc_penalty=1))
        self.assertTrue(np.isinf(m._transitions(a,b,10,100)[0,0]))
    def test_invalid_penalty_rejected(self):
        for x in (-1,float('nan'),True):
            with self.assertRaises(ValueError):ArcPolicy(appended_arc_penalty=x)
if __name__=='__main__':unittest.main()
