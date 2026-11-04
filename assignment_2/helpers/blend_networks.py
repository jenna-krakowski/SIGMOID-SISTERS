"""Select a blend of saved neural networks using validation data only.
Never reads test rows. A blend is retained only if its validation MSE improves
on the best single network. Weights are nonnegative and sum to one.
Run after improve_network.py train, before improve_network.py evaluate.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ.setdefault('MPLCONFIGDIR','/tmp/sigmoid-matplotlib')
import json
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.optimize import minimize
from sklearn.metrics import r2_score, mean_squared_error
from improve_network import OUT, ROOT, MODELS, inputs, load_splits, predict, load

def main():
    scaler,model,current=load(use_blend=False);del model;gc.collect()
    data=pd.read_csv(ROOT/'pricing.csv'); idx=load_splits(len(data))['validation']
    xv=inputs(data.iloc[idx],scaler); y=data.iloc[idx].quantity.to_numpy(dtype=float)
    results=pd.read_csv(OUT/'validation_results.csv')
    eligible=results[(results.embedding_dim>0)|(results.get('numeric_sku',False)==True)]
    eligible=eligible.sort_values('val_rmse').head(6)
    names=eligible.name.tolist();predictions=[]
    for name in names:
        model=tf.keras.models.load_model(MODELS/f'{name}.keras')
        predictions.append(predict(model,xv));del model;gc.collect()
    matrix=np.column_stack(predictions)
    variance=np.var(y)
    def objective(w):return np.mean((matrix@w-y)**2)/variance
    def gradient(w):return 2*matrix.T@(matrix@w-y)/(len(y)*variance)
    fit=minimize(objective,np.full(len(names),1/len(names)),jac=gradient,
                 bounds=[(0,1)]*len(names),constraints={'type':'eq','fun':lambda w:w.sum()-1,
                 'jac':lambda w:np.ones(len(w))},method='SLSQP',options={'ftol':1e-12,'maxiter':500})
    if not fit.success:raise RuntimeError(fit.message)
    weights=fit.x/fit.x.sum();p=matrix@weights
    result={'members':dict(zip(names,weights.tolist())),
            'validation_r2':float(r2_score(y,p)),'validation_rmse':float(mean_squared_error(y,p)**.5),
            'selection':'Convex blend fitted on validation rows; test rows not read.',
            'use_blend':bool(mean_squared_error(y,p)**.5<current['selected']['val_rmse']-1e-5)}
    (OUT/'blend.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
