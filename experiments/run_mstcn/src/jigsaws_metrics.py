import json
import numpy as np
import os

def levenstein_(p,y, norm=False):
    m_row = len(p)    
    n_col = len(y)
    D = np.zeros([m_row+1, n_col+1], float)
    for i in range(m_row+1):
        D[i,0] = i
    for i in range(n_col+1):
        D[0,i] = i

    for j in range(1, n_col+1):
        for i in range(1, m_row+1):
            if y[j-1]==p[i-1]:
                D[i,j] = D[i-1,j-1] 
            else:
                D[i,j] = min(D[i-1,j]+1,
                             D[i,j-1]+1,
                             D[i-1,j-1]+1)
    
    if norm:
        score = (1 - D[-1,-1]/max(m_row, n_col) ) * 100
    else:
        score = D[-1,-1]

    return score

def segment_labels(Yi):
    idxs = [0] + (np.nonzero(np.diff(Yi))[0]+1).tolist() + [len(Yi)]
    Yi_split = np.array([Yi[idxs[i]] for i in range(len(idxs)-1)])
    return Yi_split

def segment_intervals(Yi):
    idxs = [0] + (np.nonzero(np.diff(Yi))[0]+1).tolist() + [len(Yi)]
    intervals = [(idxs[i],idxs[i+1]) for i in range(len(idxs)-1)]
    return intervals

def get_edit_score_colin(P, Y, norm=True, bg_class=None, **kwargs):
    if type(P) == list:
        tmp = [get_edit_score_colin(P[i], Y[i], norm, bg_class)
                 for i in range(len(P))]
        return np.mean(tmp)
    else:
        P_ = segment_labels(P)
        Y_ = segment_labels(Y)
        if bg_class is not None:
            P_ = [c for c in P_ if c!=bg_class]
            Y_ = [c for c in Y_ if c!=bg_class]
        return levenstein_(P_, Y_, norm)

def get_accuracy_colin(P, Y, **kwargs):  # Average acc
    def acc_(p,y):
        return np.mean(p==y)*100
    if type(P) == list:
        return np.mean([np.mean(P[i]==Y[i]) for i in range(len(P))])*100
    else:
        return acc_(P,Y)

def get_overlap_f1_colin(P, Y, n_classes=0, bg_class=None, overlap=.1, **kwargs):
    def overlap_(p,y, n_classes, bg_class, overlap):

        true_intervals = np.array(segment_intervals(y))
        true_labels = segment_labels(y)
        pred_intervals = np.array(segment_intervals(p))
        pred_labels = segment_labels(p)

        if bg_class is not None:
            true_intervals = true_intervals[true_labels!=bg_class]
            true_labels = true_labels[true_labels!=bg_class]
            pred_intervals = pred_intervals[pred_labels!=bg_class]
            pred_labels = pred_labels[pred_labels!=bg_class]

        n_true = true_labels.shape[0]
        n_pred = pred_labels.shape[0]

        TP = np.zeros(n_classes, float)
        FP = np.zeros(n_classes, float)
        true_used = np.zeros(n_true, float)

        for j in range(n_pred):
            intersection = np.minimum(pred_intervals[j,1], true_intervals[:,1]) - np.maximum(pred_intervals[j,0], true_intervals[:,0])
            union = np.maximum(pred_intervals[j,1], true_intervals[:,1]) - np.minimum(pred_intervals[j,0], true_intervals[:,0])
            IoU = (intersection / union)*(pred_labels[j]==true_labels)

            idx = IoU.argmax()

            if IoU[idx] >= overlap and not true_used[idx]:
                TP[pred_labels[j]] += 1
                true_used[idx] = 1
            else:
                FP[pred_labels[j]] += 1


        TP = TP.sum()
        FP = FP.sum()
        FN = n_true - true_used.sum()
        
        precision = TP / (TP+FP)
        recall = TP / (TP+FN)
        F1 = 2 * (precision*recall) / (precision+recall)  #RuntimeWarning: invalid value encountered in double_scalars

        F1 = np.nan_to_num(F1)

        return F1*100

    if type(P) == list:
        return np.mean([overlap_(P[i],Y[i], n_classes, bg_class, overlap) for i in range(len(P))])
    else:
        return overlap_(P, Y, n_classes, bg_class, overlap)

def get_jg_metrics(JSON_PATH, RESULTS_SAVE_PATH, n_classes):
    with open(JSON_PATH, 'r') as f:
        output = json.load(f)
    preds = list(map(np.array, [item['prediction'] for item in output.values()]))
    gts = list(map(np.array, [item['ground_truth'] for item in output.values()]))

    with open(os.path.join(RESULTS_SAVE_PATH), 'w') as file:
        file.write(f"Accuracy: {get_accuracy_colin(preds, gts)} \n")
        file.write(f"Score: {get_edit_score_colin(preds, gts, bg_class=None)} \n")
        file.write(f"F1@10: {get_overlap_f1_colin(preds, gts, n_classes=n_classes, bg_class=None, overlap=.1)} \n")
        file.write(f"F1@25: {get_overlap_f1_colin(preds, gts, n_classes=n_classes, bg_class=None, overlap=.25)} \n")
        file.write(f"F1@50: {get_overlap_f1_colin(preds, gts, n_classes=n_classes, bg_class=None, overlap=.5)} \n")

    print("Accuracy:", get_accuracy_colin(preds, gts))
    print("Score:", get_edit_score_colin(preds, gts, bg_class=None))
    print("F1@10:", get_overlap_f1_colin(preds, gts, n_classes=n_classes, bg_class=None, overlap=.1))
    print("F1@25:", get_overlap_f1_colin(preds, gts, n_classes=n_classes, bg_class=None, overlap=.25))
    print("F1@50:", get_overlap_f1_colin(preds, gts, n_classes=n_classes, bg_class=None, overlap=.5))

def get_jg_metrics2(output, n_classes):
    preds = list(map(np.array, [item['prediction'] for item in output.values()]))
    gts = list(map(np.array, [item['ground_truth'] for item in output.values()]))
    results = {
        "accu": get_accuracy_colin(preds, gts),
        "edit": get_edit_score_colin(preds, gts, bg_class=None),
        "F1@0.1": get_overlap_f1_colin(preds, gts, n_classes=n_classes, bg_class=None, overlap=.1),
        "F1@0.25": get_overlap_f1_colin(preds, gts, n_classes=n_classes, bg_class=None, overlap=.25),
        "F1@0.5": get_overlap_f1_colin(preds, gts, n_classes=n_classes, bg_class=None, overlap=.5)
    }
    return results