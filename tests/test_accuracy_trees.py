"""Data-free bounded central menu and unchanged-family checks."""
import unittest
from sklearn.ensemble import BaggingClassifier,HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from scripts import accuracy_common as c,accuracy_trees as t

class AccuracyMenuTests(unittest.TestCase):
    def test_every_family_six_predeclared_configs_and_single_control(self):
        for family in c.FAMILIES:
            configs=t.menu(family)
            self.assertEqual(len(configs),6)
            self.assertEqual(len({x["run_name"] for x in configs}),6)
            self.assertEqual(sum(x["profile"] in ("v1","baseline38") for x in configs),1)
    def test_tree_factories_are_genuine_prescribed_families(self):
        b=t.tree_menu("bagging")[0];e=t.make_bagging_estimator(b["parameters"],20261118,1)
        self.assertIsInstance(e,BaggingClassifier);self.assertIsInstance(e.estimator,DecisionTreeClassifier);self.assertTrue(e.bootstrap)
        h=t.tree_menu("hist_gradient_boosting")[0];e=t.make_histgb_estimator(h["parameters"],20261118,1)
        self.assertIsInstance(e,HistGradientBoostingClassifier);self.assertFalse(e.early_stopping)
    def test_menus_not_mutated_by_factories(self):
        for family in ("bagging","hist_gradient_boosting"):
            item=t.tree_menu(family)[0];before=dict(item["parameters"]);t.FACTORIES[family](item["parameters"],20261118,1);self.assertEqual(item["parameters"],before)
    def test_knn_factory_genuine_inverse_distance(self):
        x=t.menu("knn_distance_weighted")[1];e=t.make_knn_estimator(x["parameters"],20261118,1);self.assertEqual(e.steps[-1][1].weights,"distance");self.assertEqual(e.steps[-1][1].n_neighbors,5)
if __name__=="__main__":unittest.main()
