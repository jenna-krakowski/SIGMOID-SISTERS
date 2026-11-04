"""Final Assignment 2 code. Run from any folder with the project .venv.
Tuning uses the saved train/validation rows only. Test evaluation is a separate command.
  .venv/bin/python assignment_2/improve_network.py train
  .venv/bin/python assignment_2/improve_network.py blend
  .venv/bin/python assignment_2/improve_network.py evaluate
  .venv/bin/python assignment_2/improve_network.py plot
  .venv/bin/python assignment_2/improve_network.py check
  .venv/bin/python assignment_2/improve_network.py predict input.csv output.csv
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ.setdefault('MPLCONFIGDIR','/tmp/sigmoid-matplotlib')
import argparse
import gc
import json
import runpy
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OUT=HERE/'results'
OUT.mkdir(exist_ok=True)
MODELS=OUT/'models'
HISTORY=OUT/'training_history'
GRAPHS=OUT/'graphs'
PREPROCESSING=OUT/'preprocessing'
for directory in [MODELS,HISTORY,GRAPHS,PREPROCESSING]:directory.mkdir(exist_ok=True)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(2)
TARGET_SCALE=50.0  # Same MSE objective, more manageable gradient magnitudes.
NUMERIC=['price','order','duration']

def load_splits(row_count):
    path=PREPROCESSING/'split_indices_70_15_15.npz'
    if path.exists():
        with np.load(path) as saved:result={key:saved[key] for key in ['train','validation','test']}
    else:
        train,remainder=train_test_split(np.arange(row_count),train_size=.70,random_state=42)
        validation,test=train_test_split(remainder,test_size=.5,random_state=42)
        result=dict(train=train,validation=validation,test=test)
        np.savez(path,**result)
    expected={'train':int(.70*row_count),'validation':int(.15*row_count),'test':int(.15*row_count)}
    assert {key:len(value) for key,value in result.items()}==expected,'Unexpected split sizes'
    assert np.array_equal(np.sort(np.concatenate(list(result.values()))),np.arange(row_count)), 'Overlapping, invalid, or missing split rows'
    (OUT/'split_manifest.json').write_text(json.dumps({'rows':row_count,'counts':expected,'random_state':42},indent=2))
    return result

def numeric_features(data):
    raw=data[NUMERIC].to_numpy(dtype=np.float64)
    if not np.isfinite(raw).all() or (raw<0).any():
        raise ValueError('price, order, and duration must be finite and nonnegative')
    # Keep raw values as well: log transforms alone can hide large-sale episodes.
    return np.column_stack([raw,np.log1p(raw)])

def inputs(data, scaler):
    return {'numeric':scaler.transform(numeric_features(data)).astype('float32'),
            'category':data.category.to_numpy(dtype='int32').reshape(-1,1),
            'sku':data.sku.to_numpy(dtype='int32').reshape(-1,1)}

def subset(x,idx): return {key:value[idx] for key,value in x.items()}

def dataset(x,y,batch,train=False,seed=42):
    ds=tf.data.Dataset.from_tensor_slices((x,y))
    if train: ds=ds.shuffle(len(y),seed=seed,reshuffle_each_iteration=True)
    options=tf.data.Options(); options.threading.private_threadpool_size=1
    return ds.batch(batch).with_options(options).prefetch(1)

def build(config, categories, skus, seed):
    tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(seed)
    numeric=tf.keras.Input((6,),name='numeric')
    category=tf.keras.Input((1,),dtype='int32',name='category')
    sku=tf.keras.Input((1,),dtype='int32',name='sku')
    cat=tf.keras.layers.IntegerLookup(vocabulary=categories,output_mode='one_hot',name='category_encoding')(category)
    pieces=[numeric,cat]
    # Connect all inputs, including SKU in no-embedding runs, without treating IDs as magnitudes.
    dim=config['embedding_dim']
    if dim:
        sku_index=tf.keras.layers.IntegerLookup(vocabulary=skus,name='sku_lookup')(sku)
        emb=tf.keras.layers.Embedding(len(skus)+1,dim,
            embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=.02),
            embeddings_regularizer=tf.keras.regularizers.L2(config.get('embedding_l2',1e-6)),
            name='sku_embedding')(sku_index)
        if not config.get('sku_effect',False):pieces.append(tf.keras.layers.Flatten()(emb))
    else:
        pieces.append(tf.keras.layers.Rescaling(scale=0.,name='sku_disabled')(sku))
    if config.get('numeric_sku',False):
        # Explicit comparison to Assignment 1's numeric-ID representation;
        # SKU ordering is a dataset-specific proxy, not a semantic distance.
        pieces.append(tf.keras.layers.Rescaling(scale=1/75000.,name='sku_numeric')(sku))
    h=tf.keras.layers.Concatenate()(pieces)
    # deep.pdf slides 69-73: He initialization for ReLU-family activations.
    # Slides 77, 80-90: compare ELU and batch normalization.
    for width in config['layers']:
        h=tf.keras.layers.Dense(width,kernel_initializer=config.get('initializer','glorot_uniform'),
            use_bias=not config.get('batch_norm',False),
            kernel_regularizer=tf.keras.regularizers.L2(config['l2']))(h)
        if config.get('batch_norm',False):
            h=tf.keras.layers.BatchNormalization()(h)
        h=tf.keras.layers.Activation(config.get('activation','relu'))(h)
        if config.get('dropout',0):h=tf.keras.layers.Dropout(config['dropout'])(h)
    output=tf.keras.layers.Dense(1,name='scaled_quantity',kernel_initializer=config.get('output_initializer','glorot_uniform'))(h)
    if config.get('sku_effect',False):
        # A single regularized product offset has much less capacity to memorize
        # noise than an unconstrained multi-dimensional product embedding.
        output=tf.keras.layers.Add()([output,tf.keras.layers.Flatten()(emb)])
    if config.get('duration_rate',False):
        # Quantity = predicted sales rate * observed duration. Still trained with
        # quantity MSE; this changes the parameterization, not the evaluation target.
        duration=tf.keras.layers.Rescaling(config['duration_scale'],config['duration_mean'])(numeric[:,2:3])
        if config.get('duration_power',1.)!=1.:
            duration=tf.keras.ops.power(tf.keras.ops.maximum(duration+config.get('duration_offset',0.),.01),config['duration_power'])
        output=tf.keras.layers.Multiply()([output,duration])
    model=tf.keras.Model({'numeric':numeric,'category':category,'sku':sku},output)
    # deep.pdf slides 96-99: clipping; 117-118: decoupled AdamW decay.
    optimizer=(tf.keras.optimizers.AdamW(config['lr'],weight_decay=config.get('weight_decay',1e-4),clipnorm=config.get('clipnorm',5.))
               if config.get('optimizer')=='adamw' else tf.keras.optimizers.Adam(config['lr'],clipnorm=config.get('clipnorm',5.)))
    model.compile(optimizer=optimizer,loss='mse',metrics=[tf.keras.metrics.MeanSquaredError(name='prediction_mse')],
                  steps_per_execution=16)
    return model

def predict(model,x):
    return np.concatenate([model({k:v[i:i+4096] for k,v in x.items()},training=False).numpy().ravel()
                           for i in range(0,len(x['numeric']),4096)])*TARGET_SCALE

class Progress(tf.keras.callbacks.Callback):
    def __init__(self,name): super().__init__(); self.name=name
    def on_epoch_end(self,epoch,logs=None):
        if epoch==0 or (epoch+1)%10==0:
            message=f"{self.name} epoch {epoch+1}: train RMSE={np.sqrt(logs.get('prediction_mse',logs['loss']))*TARGET_SCALE:.3f}"
            if 'val_loss' in logs:message+=f", validation RMSE={np.sqrt(logs.get('val_prediction_mse',logs['val_loss']))*TARGET_SCALE:.3f}"
            print(message,flush=True)

def train(group='best'):
    data=pd.read_csv(ROOT/'pricing.csv'); split=load_splits(len(data))
    tr,va=split['train'],split['validation']
    scaler=StandardScaler().fit(numeric_features(data.iloc[tr]))
    np.savez(PREPROCESSING/'preprocessing.npz',mean=scaler.mean_,scale=scaler.scale_)
    x=inputs(data,scaler); y=data.quantity.to_numpy(dtype='float32')
    categories=sorted(data.iloc[tr].category.unique().astype(int).tolist())
    skus=sorted(data.iloc[tr].sku.unique().astype(int).tolist())
    initial_configs=[
        dict(name='category_wide',layers=[128,64,32],embedding_dim=0,l2=1e-6,lr=.001,batch=512),
        dict(name='category_larger',layers=[256,128,64],embedding_dim=0,l2=1e-5,lr=.001,batch=512),
        dict(name='sku_embedding',layers=[128,64,32],embedding_dim=4,l2=1e-5,embedding_l2=1e-5,lr=.001,batch=512),
    ]
    # Direct applications of deep.pdf: embeddings (4-5), ELU (77), He (73),
    # BatchNorm (80-90), AdamW (117-118), depth vs width (133-135).
    lecture_configs=[
            dict(name='lecture_elu_bn',layers=[128,64,32],embedding_dim=0,l2=0.,lr=.001,batch=512,
                 initializer='he_normal',activation='elu',batch_norm=True,optimizer='adamw'),
            dict(name='lecture_embedding_elu',layers=[128,64,32],embedding_dim=4,l2=0.,embedding_l2=1e-5,
                 lr=.001,batch=512,initializer='he_normal',activation='elu',optimizer='adamw'),
            dict(name='lecture_embedding_bn',layers=[256,128,64],embedding_dim=8,l2=0.,embedding_l2=1e-5,
                 lr=.001,batch=512,initializer='he_normal',activation='elu',batch_norm=True,optimizer='adamw'),
        ]
    refinement_configs=[
            dict(name='unclipped_relu',layers=[256,128,64],embedding_dim=0,l2=0.,lr=.001,batch=512,
                 initializer='he_normal',activation='relu',clipnorm=None,numeric_sku=True),
            dict(name='regularized_embedding',layers=[128,64,32],embedding_dim=2,l2=1e-5,embedding_l2=.001,
                 lr=.001,batch=512,initializer='he_normal',activation='elu',clipnorm=None),
            dict(name='deep_elu',layers=[256,128,64,32,16],embedding_dim=0,l2=1e-6,lr=.0005,batch=256,
                 initializer='he_normal',activation='elu',clipnorm=None,numeric_sku=True),
        ]
    rate_configs=[
        dict(name='sku_rate_model',layers=[128,64,32],embedding_dim=1,sku_effect=True,
             embedding_l2=1e-4,l2=1e-5,lr=.001,batch=512,initializer='he_normal',activation='elu',
             duration_rate=True,duration_scale=float(scaler.scale_[2]),duration_mean=float(scaler.mean_[2]),clipnorm=None),
        dict(name='sku_offset_model',layers=[128,64,32],embedding_dim=1,sku_effect=True,
             embedding_l2=3e-5,l2=1e-5,lr=.001,batch=512,initializer='he_normal',activation='elu',clipnorm=None),
    ]
    ensemble_configs=[
        dict(name='rate_seed_123',seed=123,layers=[128,64,32],embedding_dim=1,sku_effect=True,
             embedding_l2=1e-4,l2=1e-5,lr=.001,batch=512,initializer='he_normal',activation='elu',
             duration_rate=True,duration_scale=float(scaler.scale_[2]),duration_mean=float(scaler.mean_[2]),clipnorm=None),
        dict(name='rate_seed_2026',seed=2026,layers=[128,64,32],embedding_dim=1,sku_effect=True,
             embedding_l2=1e-4,l2=1e-5,lr=.001,batch=512,initializer='he_normal',activation='elu',
             duration_rate=True,duration_scale=float(scaler.scale_[2]),duration_mean=float(scaler.mean_[2]),clipnorm=None),
        dict(name='rate_dropout',seed=42,layers=[128,64,32],embedding_dim=1,sku_effect=True,
             embedding_l2=3e-4,l2=1e-5,lr=.001,batch=512,initializer='he_normal',activation='elu',dropout=.1,
             duration_rate=True,duration_scale=float(scaler.scale_[2]),duration_mean=float(scaler.mean_[2]),clipnorm=None),
    ]
    power_configs=[
        dict(name=f'duration_power_{power}',layers=[128,64,32],embedding_dim=0,numeric_sku=True,
             l2=1e-5,lr=.0005,batch=512,initializer='he_normal',activation='elu',
             duration_rate=True,duration_power=power,duration_offset=-.9,output_initializer='zeros',
             duration_scale=float(scaler.scale_[2]),duration_mean=float(scaler.mean_[2]),clipnorm=5.)
        for power in [1.5,2.]
    ]
    groups={'initial':initial_configs,'lecture':lecture_configs,'refinement':refinement_configs,
            'rate':rate_configs,'ensemble':ensemble_configs,'power':power_configs}
    groups['best']=[refinement_configs[2]]+rate_configs+ensemble_configs
    configs=[config for key,batch in groups.items() if key!='best' for config in batch] if group=='all' else groups[group]
    results=pd.read_csv(OUT/'validation_results.csv').to_dict('records') if (OUT/'validation_results.csv').exists() else []
    trained_new_model=False
    for config in configs:
        if any(row['name']==config['name'] for row in results) and (MODELS/f"{config['name']}.keras").exists():
            print('Already completed:',config['name'],flush=True)
            continue
        results=[row for row in results if row['name']!=config['name']]
        seed=config.get('seed',42)
        model=build(config,categories,skus,seed)
        # Monitor prediction MSE rather than MSE + regularization penalties.
        callbacks=[tf.keras.callbacks.ReduceLROnPlateau(monitor='val_prediction_mse',factor=.5,patience=6,min_lr=1e-5),
                   tf.keras.callbacks.EarlyStopping(monitor='val_prediction_mse',patience=18,restore_best_weights=True),
                   tf.keras.callbacks.TerminateOnNaN(),Progress(config['name'])]
        start=time.perf_counter()
        history=model.fit(dataset(subset(x,tr),y[tr]/TARGET_SCALE,config['batch'],True,seed),
                          validation_data=dataset(subset(x,va),y[va]/TARGET_SCALE,4096),
                          epochs=150,callbacks=callbacks,shuffle=False,verbose=0).history
        seconds=time.perf_counter()-start
        p=predict(model,subset(x,va))
        row={**config,'seed':seed,'val_r2':float(r2_score(y[va],p)),
             'val_rmse':float(mean_squared_error(y[va],p)**.5),'epochs':len(history['loss']),
             'best_epoch':int(np.argmin(history['val_prediction_mse'])+1),'seconds':seconds,'parameters':model.count_params()}
        model.save(MODELS/f"{config['name']}.keras")
        trained_new_model=True
        (HISTORY/f"history_{config['name']}.json").write_text(json.dumps(history,indent=2))
        results.append(row);pd.DataFrame(results).to_csv(OUT/'validation_results.csv',index=False)
        print('RESULT',json.dumps(row),flush=True)
        del model;gc.collect()
    # User requested that SKU remain in the final model; no-SKU models are ablations.
    eligible=[row for row in results if row.get('embedding_dim',0)>0 or row.get('numeric_sku',False)==True]
    selected=min(eligible,key=lambda row:row['val_rmse'])
    selected={key:(None if isinstance(value,float) and not np.isfinite(value) else value)
              for key,value in selected.items()}
    if isinstance(selected.get('layers'),str):selected['layers']=json.loads(selected['layers'])
    (OUT/'selected.json').write_text(json.dumps({'selected':selected,'selection':'lowest validation RMSE among models using SKU; test not used',
            'target_scale':TARGET_SCALE,'train_rows':len(tr),'validation_rows':len(va)},indent=2))
    print('SELECTED',json.dumps(selected),flush=True)
    if trained_new_model and (OUT/'blend.json').exists():
        blend=json.loads((OUT/'blend.json').read_text())
        blend['use_blend']=False
        blend['selection']='New models trained; run improve_network.py blend before using a blend.'
        (OUT/'blend.json').write_text(json.dumps(blend,indent=2))

class NeuralBlend:
    """Inference-only weighted average of the saved feed-forward networks."""
    def __init__(self,members):self.members=members
    def __call__(self,x,training=False):
        return tf.add_n([weight*model(x,training=False) for weight,model in self.members])

def load(use_blend=True):
    saved=np.load(PREPROCESSING/'preprocessing.npz'); scaler=StandardScaler()
    scaler.mean_=saved['mean'];scaler.scale_=saved['scale'];scaler.n_features_in_=6
    chosen=json.loads((OUT/'selected.json').read_text())
    model=tf.keras.models.load_model(MODELS/f"{chosen['selected']['name']}.keras")
    if use_blend and (OUT/'blend.json').exists():
        blend=json.loads((OUT/'blend.json').read_text())
        if blend['use_blend']:
            model=NeuralBlend([(weight,tf.keras.models.load_model(MODELS/f'{name}.keras'))
                              for name,weight in blend['members'].items() if weight>1e-8])
            chosen['blend']=blend
    return scaler,model,chosen

def evaluate():
    scaler,model,chosen=load()
    data=pd.read_csv(ROOT/'pricing.csv'); splits=load_splits(len(data))
    x=inputs(data,scaler);y=data.quantity.to_numpy(dtype='float32');metrics={}
    for name,idx in splits.items():
        p=predict(model,subset(x,idx))
        metrics[name]={'r2':float(r2_score(y[idx],p)),'rmse':float(mean_squared_error(y[idx],p)**.5)}
        if name=='test':pd.DataFrame({'row':idx,'actual':y[idx],'prediction':p}).to_csv(OUT/'test_predictions.csv',index=False)
    (OUT/'metrics.json').write_text(json.dumps({'configuration':chosen['selected'],'blend':chosen.get('blend'),'metrics':metrics,
            'split':'70% training / 15% validation / 15% test',
            'test_note':'Split changed after earlier experiments; some rows overlap earlier evaluations. Test targets not used in this search.'},indent=2))
    pd.DataFrame([{'set':name,'rows':len(splits[name]),**values}
                  for name,values in metrics.items()]).to_csv(OUT/'FINAL_SCORES.csv',index=False)
    print(json.dumps(metrics,indent=2))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['train','blend','evaluate','plot','check','predict'],nargs='?',default='train')
    parser.add_argument('input_csv',nargs='?');parser.add_argument('output_csv',nargs='?')
    parser.add_argument('--group',choices=['all','initial','lecture','refinement','rate','ensemble','power','best'],default='best',
                        help='Run a subset of experiments; completed runs are skipped automatically')
    args=parser.parse_args()
    if args.action=='train':train(args.group)
    elif args.action=='evaluate':evaluate()
    elif args.action in ['blend','plot','check']:
        helper={'blend':'blend_networks.py','plot':'plot_improved.py','check':'validate_results.py'}[args.action]
        runpy.run_path(str(HERE/'helpers'/helper),run_name='__main__')
    else:
        if not args.input_csv or not args.output_csv:parser.error('predict requires input and output CSV paths')
        scaler,model,_=load();data=pd.read_csv(args.input_csv)
        data['predicted_quantity']=predict(model,inputs(data,scaler));data.to_csv(args.output_csv,index=False)

if __name__=='__main__':
    sys.modules.setdefault('improve_network',sys.modules[__name__])
    main()
