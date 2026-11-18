"""No raw-data tests for dense fold-qualified route validation."""
import unittest
from types import SimpleNamespace
import numpy as np
from scripts import accuracy_route_validation as r

class DenseRouteValidationTests(unittest.TestCase):
    def test_hash_plan_wholetrip_stable_and_excludes_fitting(self):
        ids=[f"{i:08d}" for i in range(18)]
        folds=[{"fold":i,"validation_trip_ids":ids[i*6:i*6+6],"train_trip_ids":ids[:i*6]+ids[i*6+6:]} for i in range(3)]
        out=r.selected_routes({"folds":folds});self.assertEqual(len(out),6);self.assertEqual(out,r.selected_routes({"folds":folds}))
        for x in out:self.assertNotIn(x["trip_id"],x["train_trip_ids"])
    def test_emission_reuse_requires_exact_track_and_candidate_geometry(self):
        class Model:
            calls=0
            def score_candidates_batch(self,g,t,i,c):self.calls+=1;return [np.array([.6,.4])]
        m=Model();e=r.CachedEmission(m);c=SimpleNamespace(edges=np.array([0,1]),fractions=np.array([.1,.2]),distances=np.array([1.,2.]));t=np.array([[1.,2.,0.]])
        p=e.score_candidates_batch(None,t,[0],[c]);p[0][0]=0.;again=e.score_candidates_batch(None,t,[0],[c]);self.assertEqual(m.calls,1);self.assertEqual(again[0][0],.6)
        c.fractions[0]=.2
        with self.assertRaises(ValueError):e.score_candidates_batch(None,t,[0],[c])
    def test_initial_decoder_menu_must_remain_dense_fixedK(self):
        m={"configurations":[{"name":"b","match_overrides":{},"transition_policy":{}}]}
        c=r.configurations(m)[0]["config"];self.assertEqual(c.max_candidates,12);self.assertEqual(c.observation_distance_m,0);self.assertEqual(c.max_radius_m,1000)
        m["configurations"][0]["match_overrides"]={"observation_distance_m":10}
        with self.assertRaises(ValueError):r.configurations(m)
    def test_only_predetermined_tuning_trip_can_be_loaded(self):
        with self.assertRaises(ValueError):r.trip_prefix("notallowed",["allowed"])
if __name__=="__main__":unittest.main()
