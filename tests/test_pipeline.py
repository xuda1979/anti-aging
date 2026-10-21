"""Focused scientific contract tests; run python -m unittest discover -s tests."""
import tempfile
import unittest
from pathlib import Path
import joblib
import numpy as np
from rejuvenation.pipeline import TrainingFeatures, frozen_split, parse_age, read_dataset, ridge_pipeline

ROOT=Path(__file__).resolve().parents[1]

class PipelineContracts(unittest.TestCase):
    def test_geo_age_parser(self):
        self.assertEqual(parse_age("96"),96)
        self.assertEqual(parse_age("2yr3mos"),2.25)
        with self.assertRaises(ValueError): parse_age("title_19yr_Male")

    def test_feature_selection_fit_is_train_only(self):
        train=np.array([[0.,6.,10.],[0.,8.,15.],[0.,7.,11.]])
        ft=TrainingFeatures(max_features=1).fit(train)
        chosen=ft.selected_indices_.copy()
        holdout=np.array([[1e9,0.,0.]])
        ft.transform(holdout)
        np.testing.assert_array_equal(chosen,ft.selected_indices_)
        self.assertNotEqual(chosen[0],0)
        np.testing.assert_allclose(ft.transform(train),np.log1p(train[:,chosen]))

    def test_invalid_measurements_fail_closed(self):
        with self.assertRaises(ValueError): TrainingFeatures().fit([[np.nan,1.]])
        with self.assertRaises(ValueError): TrainingFeatures().fit([[-1.,6.]])
        ft=TrainingFeatures().fit([[6.,7.],[8.,9.]])
        with self.assertRaises(ValueError): ft.transform([[1.,2.,3.]])

    def test_real_schema_split_and_checkpoint(self):
        X,meta,annotation=read_dataset(ROOT/'data/raw')
        train,test=frozen_split(meta)
        self.assertEqual(X.shape,(143,27142));self.assertEqual(len(annotation),27142)
        self.assertFalse(set(meta.iloc[train].cell_id)&set(meta.iloc[test].cell_id))
        self.assertTrue(meta.iloc[np.r_[train,test]].eligible.all())
        self.assertTrue((meta.iloc[np.r_[train,test]].age_years>=18).all())
        train2,test2=frozen_split(meta)
        np.testing.assert_array_equal(train,train2);np.testing.assert_array_equal(test,test2)
        model=ridge_pipeline(10).fit(X[train[:12]],meta.iloc[train[:12]].age_years)
        pred=model.predict(X[test[:3]])
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'model.joblib';joblib.dump(model,path)
            np.testing.assert_allclose(joblib.load(path).predict(X[test[:3]]),pred,rtol=0,atol=1e-12)

if __name__=='__main__': unittest.main()
