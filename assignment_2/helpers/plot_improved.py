"""Graph the revised network's recorded results without retraining."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/sigmoid-matplotlib')
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from improve_network import OUT, HISTORY, GRAPHS

def main():
    results=pd.read_csv(OUT/'validation_results.csv').sort_values('val_r2')
    selected=json.loads((OUT/'selected.json').read_text())['selected']
    history=json.loads((HISTORY/f"history_{selected['name']}.json").read_text())
    if (OUT/'blend.json').exists():
        blend=json.loads((OUT/'blend.json').read_text())
        if blend['use_blend']:
            results=pd.concat([results,pd.DataFrame([{'name':'Neural-network blend','val_r2':blend['validation_r2']}])]).sort_values('val_r2')
    plt.figure(figsize=(10,6))
    plt.barh(results.name,results.val_r2,color=['#497986' if r['name']!=selected['name'] else '#ca733d' for _,r in results.iterrows()])
    plt.axvline(.65,color='black',ls='--',label='Requested R² = 0.65')
    plt.xlabel('Validation R² (higher is better)');plt.title('Revised neural-network comparisons')
    plt.legend();plt.tight_layout();plt.savefig(GRAPHS/'validation_comparison.png',dpi=150);plt.close()
    plt.figure(figsize=(9,5))
    for key,label in [('prediction_mse','Training'),('val_prediction_mse','Validation')]:
        plt.plot(np.arange(1,len(history[key])+1),np.sqrt(history[key])*50,label=label)
    plt.axvline(selected['best_epoch'],ls='--',color='gray',label='Best validation epoch')
    plt.xlabel('Epoch');plt.ylabel('Quantity RMSE');plt.title('Best single network: '+selected['name'])
    plt.legend();plt.tight_layout();plt.savefig(GRAPHS/'selected_learning_curve.png',dpi=150);plt.close()
    if (OUT/'test_predictions.csv').exists():
        p=pd.read_csv(OUT/'test_predictions.csv')
        metrics=json.loads((OUT/'metrics.json').read_text())['metrics']['test']
        plt.figure(figsize=(7,5))
        plt.hexbin(p.actual,p.prediction,gridsize=55,bins='log',mincnt=1,cmap='viridis')
        limit=max(p.actual.max(),p.prediction.max());plt.plot([0,limit],[0,limit],'r--',lw=1)
        plt.xlabel('Actual quantity');plt.ylabel('Predicted quantity');plt.colorbar(label='log10(count)')
        plt.title(f"75,000 test rows — R² = {metrics['r2']:.4f}")
        plt.tight_layout();plt.savefig(GRAPHS/'test_predictions.png',dpi=150);plt.close()
    print('Saved improved model comparison, learning curve, and available test graph.')

if __name__=='__main__':main()
