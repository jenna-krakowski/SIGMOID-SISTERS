"""Audit the 70/15/15 split, training-only preprocessing, and saved predictions."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score,mean_squared_error
from improve_network import OUT,ROOT,PREPROCESSING,load_splits,numeric_features,inputs,load,predict

def main():
    data=pd.read_csv(ROOT/'pricing.csv'); split=load_splits(len(data))
    assert [len(split[key]) for key in ['train','validation','test']]==[350000,75000,75000]
    scaled=np.load(PREPROCESSING/'preprocessing.npz')
    values=numeric_features(data.iloc[split['train']])
    assert np.allclose(scaled['mean'],values.mean(axis=0))
    assert np.allclose(scaled['scale'],values.std(axis=0))
    metrics=json.loads((OUT/'metrics.json').read_text())
    saved=pd.read_csv(OUT/'test_predictions.csv')
    assert np.array_equal(saved.row.to_numpy(),split['test'])
    assert np.isfinite(saved.prediction).all()
    assert np.allclose(saved.actual,data.iloc[split['test']].quantity)
    assert np.isclose(r2_score(saved.actual,saved.prediction),metrics['metrics']['test']['r2'],atol=1e-6)
    assert np.isclose(mean_squared_error(saved.actual,saved.prediction)**.5,metrics['metrics']['test']['rmse'],atol=1e-5)
    if (OUT/'blend.json').exists():
        blend=json.loads((OUT/'blend.json').read_text())
        assert np.isclose(sum(blend['members'].values()),1.)
        assert min(blend['members'].values())>=-1e-8
    scaler,model,_=load()
    new=predict(model,inputs(data.iloc[split['test'][:10]],scaler))
    assert np.allclose(new,saved.prediction.head(10),rtol=1e-5,atol=1e-4)
    result={'split':'70/15/15','counts':{key:len(value) for key,value in split.items()},
            'checks':'passed: disjoint split, training-only scaler, recomputed metrics, saved-model round trip',
            'test_r2':metrics['metrics']['test']['r2'],'target_r2':.65,
            'target_met':bool(metrics['metrics']['test']['r2']>=.65)}
    (OUT/'audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
