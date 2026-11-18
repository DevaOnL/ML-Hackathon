"""Small deterministic checks for comparison accuracy and reference-free audit."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
from mapmatch.data import Graph
from mapmatch.candidates import CandidateIndex
from mapmatch.features import FEATURE_NAMES,reference_observation_labels
from mapmatch.learning import EmissionModel
from mapmatch.matcher import MapMatcher,MatchConfig
from scripts.target96_release import FullAudit,observation_metrics

class TinyEstimator:
    classes_=np.array([0,1]);n_features_in_=len(FEATURE_NAMES)
    def predict_proba(self,X):
        positive=1/(1+np.exp(np.clip(X[:,0]/20,-30,30)))
        return np.c_[1-positive,positive]

class FinalComparisonTests(unittest.TestCase):
    def test_fullaudit_observations_binary_vs_categorical(self):
        graph=Graph(np.array([[0,0],[.001,0],[.002,0]],float),np.array([[0,1],[1,2],[1,0],[2,1]],int));track=np.array([[.0001,0,0],[.0008,0,1],[.0017,0,2]],float)
        model=EmissionModel(TinyEstimator(),{});audit=FullAudit(model);result=MapMatcher(graph,MatchConfig(max_candidates=4,candidate_retry_limit=4),audit).match(track)
        labels=reference_observation_labels(graph,track,np.array([0,1]))
        with tempfile.TemporaryDirectory() as td:
            metrics,rawroute=observation_metrics(graph,track,labels,result,audit,model,'v1',Path(td)/'observations.npz')
            self.assertEqual(metrics['observations'],3);self.assertEqual(len(audit.candidates),3)
            self.assertTrue(np.isfinite(metrics['log_loss']));self.assertTrue(np.isfinite(metrics['observation_nll']));self.assertGreaterEqual(metrics['probability_call_seconds'],0)
            self.assertEqual(metrics['candidate_miss_count'],0)
            self.assertEqual(np.load(Path(td)/'observations.npz')['raw'].shape,(metrics['candidate_rows'],))
            self.assertGreater(len(rawroute),0)
if __name__=='__main__':unittest.main()
