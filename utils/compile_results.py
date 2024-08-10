import os
import numpy as np
import json

model_list = ["mstcn"]
dataset_list = ["proposed_nets_videomae"]

def get_splitwise_results(model, dataset):
    record_dir = os.path.join('record', model, dataset)
    records = list(sorted(filter(lambda x: "_colin_best.csv" in x, os.listdir(record_dir))))
    records = list(map(lambda x: os.path.join(record_dir, x), records))
    res = []
    for f in records:
        with open(f, "r") as t:
            lines = t.readlines()
        lines = list(map(lambda x: x.replace('\n', ""), lines))
        res.append(list(map(lambda x: float(x), lines[-1].split('\t')[1:])))
    res = np.array(res)
    return np.mean(res, axis = 0).tolist()


def combine_results():
    results = {}
    for dataset in dataset_list:
        inter = {}
        for model in model_list:
            inter[model] = get_splitwise_results(model, dataset)
        results[dataset] = inter

    with open("results.json", "w") as f:
        json.dump(results, f)  
        
combine_results()  
    
    