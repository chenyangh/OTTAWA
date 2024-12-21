import pandas as pd
import pickle as pkl
import numpy as np  
import torch
from ot.backend import get_backend
import torch.nn.functional as F

speical_token_list = [":", ".", "\"", ",", "!", "?", "(", ")", "[", "]", "{", "}", "<", ">", "-", "_", "=", "+", "*", "/", "\\", "|", "@", "#", "$", "%", "^", "&", "~", "`", ";", "'", "``", "''"]

def percent_correct_pairs(y, x):
    """Given the targets y and predictions x, compute the percentage of pairs with different y that are correctly ordered by x.
    This metric is a multi-class extension of ROC AUC.
    """
    n_correct, n_total = 0, 0
    for x1, y1 in zip(x, y):
        for x2, y2 in zip(x, y):
            if y1 >= y2:
                continue
            n_total += 1
            n_correct += x1 < x2
    return n_correct / max(n_total, 1)

def print_matrix(P, x, y):
    # Round the matrix to 2 decimal places
    P_rounded = np.round(P, 2)

    # Print column names
    print("     ", "  ".join(y))

    # Print each row with its name
    for row_name, row in zip(x, P_rounded):
        row_str = '  '.join(str(val) for val in row)
        print(f"{row_name:5s}: {row_str}")
         
        
def get_data():
    import os
    from pathlib import Path

    bin_path = Path('data/hallucinations_deen_w_stats_and_scores.pkl')

    dataset_stats = pkl.load(open(bin_path, 'rb'))
    annotated = pd.read_csv('data/annotated_corpus.csv')

    merged_data = pd.merge(
        dataset_stats,
        annotated,
        how="inner",
        on=None,
        left_on=None,
        right_on=None,
        left_index=False,
        right_index=False,
        sort=True,
        suffixes=("_x", "_y"),
        copy=True,
        indicator=False,
        validate=None,
    )
    return merged_data


def min_max_scaling(C):
    eps = 1e-10
    # Min-max scaling for stabilization
    nx = get_backend(C)
    C_min = nx.min(C)
    C_max = nx.max(C)
    C = (C - C_min + eps) / (C_max - C_min + eps)
    return C

def compute_distance_matrix_cosine(s1_word_embeddigs, s2_word_embeddigs, distortion_ratio):
    C = (torch.matmul(F.normalize(s1_word_embeddigs), F.normalize(s2_word_embeddigs).t()) + 1.0) / 2  # Range 0-1
    # C = apply_distortion(C, distortion_ratio)
    # C = min_max_scaling(C)  # Range 0-1
    C = 1.0 - C  # Convert to distance
    return C

def compute_distance_matrix_l2(s1_word_embeddigs, s2_word_embeddigs, distortion_ratio):
    # s1_word_embeddigs = F.normalize(s1_word_embeddigs)
    # s2_word_embeddigs = F.normalize(s2_word_embeddigs)
    C = torch.cdist(s1_word_embeddigs, s2_word_embeddigs, p=2)
    # C = min_max_scaling(C)  # Range 0-1
    # C = 1.0 - C  # Convert to similarity
    # C = apply_distortion(C, distortion_ratio)
    # C = min_max_scaling(C)  # Range 0-1
    # C = 1.0 - C  # Convert to distance
    return C


def apply_distortion(sim_matrix, ratio):
    shape = sim_matrix.shape
    if (shape[0] < 2 or shape[1] < 2) or ratio == 0.0:
        return sim_matrix

    pos_x = torch.tensor([[y / float(shape[1] - 1) for y in range(shape[1])] for x in range(shape[0])],
                         device=sim_matrix.device)
    pos_y = torch.tensor([[x / float(shape[0] - 1) for x in range(shape[0])] for y in range(shape[1])],
                         device=sim_matrix.device)
    distortion_mask = 1.0 - ((pos_x - pos_y.T) ** 2) * ratio
    sim_matrix = torch.mul(sim_matrix, distortion_mask)
    return sim_matrix


def compute_weights_norm(s1_word_embeddigs, s2_word_embeddigs):
    s1_weights = torch.norm(s1_word_embeddigs, dim=1)
    s2_weights = torch.norm(s2_word_embeddigs, dim=1)
    return s1_weights, s2_weights


def compute_weights_uniform(s1_word_embeddigs, s2_word_embeddigs):
    s1_weights = torch.ones(s1_word_embeddigs.shape[0], dtype=s1_word_embeddigs.dtype, device=s1_word_embeddigs.device)
    s2_weights = torch.ones(s2_word_embeddigs.shape[0], dtype=s1_word_embeddigs.dtype, device=s1_word_embeddigs.device)
    return s1_weights, s2_weights


from typing import Dict
import pandas as pd
import numpy as np
import sklearn.metrics as skm


def get_fpr_tpr_thr(y_true, y_pred, pos_label):
    """Computes the FPR, TPR and THRESHOLD for a binary classification problem.
        * `y_score >= threhold` is classified as `pos_label`.
    Args:
        y_true ([type]): [description]
        y_pred ([type]): [description]
        pos_label ([type]): [description]
    Returns:
        fpr : Increasing false positive rates such that element i is the false
            positive rate of predictions with score >= `thresholds[i]`.
        tpr : Increasing true positive rates such that element `i` is the true
            positive rate of predictions with score >= `thresholds[i]`.
        thresholds : Decreasing thresholds on the decision function used to compute
            fpr and tpr. `thresholds[0]` represents no instances being predicted
            and is arbitrarily set to `max(y_score) + 1`.
    """
    # print(1)
    # y_true = y_true.astype(int)
    fpr, tpr, thresholds = skm.roc_curve(y_true, y_pred, pos_label=pos_label)
    return fpr, tpr, thresholds


def get_precision_recall_thr(y_true, y_pred, pos_label=1):
    precision, recall, thresholds = skm.precision_recall_curve(
        y_true, y_pred, pos_label=pos_label
    )
    return precision, recall, thresholds

def compute_auroc(fpr, tpr):
    return skm.auc(fpr, tpr)

def compute_fpr_tpr_thr_given_tpr_level(fpr, tpr, thresholds, tpr_level):
    if all(tpr < tpr_level):
        raise ValueError(f"No threshold allows for TPR at least {tpr_level}.")
    # elif all(tpr >= tpr_level):
    #     # All thresholds allow TPR >= tpr level, so find lowest possible FPR
    #     idx = np.argmin(fpr)
    else:
        idxs = [i for i, x in enumerate(tpr) if x >= tpr_level]
        idx = min(idxs)
    return fpr[idx], tpr[idx], thresholds[idx]

def compute_fpr_tpr(fpr, tpr, thresholds):
    idx = min(i for i, x in enumerate(tpr) if x >= 0.95)
    return fpr[idx], tpr[idx], thresholds[idx]

def compute_metrics(df_stats, category: str, metrics: list):
    eval_metrics = []
    for metric in metrics:
        detector_scores = df_stats[metric].values
        fpr, tpr, thresholds = get_fpr_tpr_thr(df_stats[category].values, detector_scores, pos_label=1)
        fpr_at_90tpr, _, _ = compute_fpr_tpr_thr_given_tpr_level(fpr, tpr, thresholds, tpr_level=0.9)
        precision, recall, thresholds = get_precision_recall_thr(df_stats[category].values, detector_scores, pos_label=1)
        auc_roc = compute_auroc(fpr, tpr)
        auc = {"metric": metric, "auc-ROC": auc_roc * 100, "fprat90tpr": fpr_at_90tpr * 100}
        eval_metrics.append(auc)
    df = pd.DataFrame(eval_metrics)
    return df.sort_values(by="auc-ROC", ascending = False)

def show_results(new_df):
    # new_df = pd.read_csv(Path(results_path, f'hallucination_{method}.csv'))
    df_copy = new_df.copy() 

    cat_list = ["is_hall", "is_osc", "is_fd", "is_sd", "strong-unsupport", "full-unsupport", "omission", "named-entities", "repetitions"]
    # 'src_log_prob_score_wt', 'src_unalign_negwt', 'src_log_prob_score', 'src_log_prob_score_wt', 'src_log_prob_score_negwt', 'tgt_unalign', 'tgt_unalign_wt', 'tgt_unalign_negwt',  'tgt_log_prob_score', 'tgt_log_prob_score_wt', 'tgt_log_prob_score_negwt', 'tgt_cost_thres0_1', 'tgt_log_prob_all'
    # cat_list = ["is_sd", "is_fd", "omission"]
    for category in ['Method'] + cat_list:
        print(category, end=' ')
    print()
    for metric in ['comet', 'seqlogprob', 'attn_ign_src', 'laser', 'chrf2', 'alti', 'cometkiwi', 'labse', 'wass_to_unif', 'wass_to_data', 'wass_combo', 'NULL_score_sum', 'NULL_score_sum_mask']:
        print(metric, end=' ')
        if metric in ['comet', 'seqlogprob', 'laser', 'chrf2', 'alti', 'cometkiwi', 'labse']:
                df_copy[metric] = - df_copy[metric]
                
        for category in cat_list:
            df_copy[category] = df_copy[category].astype(int)
            df_osc = compute_metrics(df_copy, category=category, metrics = [metric])

            auroc_osc = df_osc["auc-ROC"].values[0]
            fprat90_osc = df_osc["fprat90tpr"].values[0]

            # print(f"Category: {category}, AUROC: {auroc_osc}, FPRat90TPR: {fprat90_osc}")
            
            print(auroc_osc, end=' ')
        print()

    # for category in ["is_hall", "is_osc", "is_fd", "is_sd", "strong-unsupport", "full-unsupport", "omission", "named-entities", "repetitions"]:
    #     new_df[category] = new_df[category].astype(int)
    #     df_osc = compute_metrics(new_df, category=category, metrics = ['sig_align_count'])

    #     auroc_osc = df_osc["auc-ROC"].values[0]
    #     fprat90_osc = df_osc["fprat90tpr"].values[0]

    #     # print(f"Category: {category}, AUROC: {auroc_osc}, FPRat90TPR: {fprat90_osc}")
        
    #     print(auroc_osc, end=' ')

def get_fpr_tpr_thr(y_true, y_pred, pos_label):
    """Computes the FPR, TPR and THRESHOLD for a binary classification problem.
        * `y_score >= threhold` is classified as `pos_label`.
    Args:
        y_true ([type]): [description]
        y_pred ([type]): [description]
        pos_label ([type]): [description]
    Returns:
        fpr : Increasing false positive rates such that element i is the false
            positive rate of predictions with score >= `thresholds[i]`.
        tpr : Increasing true positive rates such that element `i` is the true
            positive rate of predictions with score >= `thresholds[i]`.
        thresholds : Decreasing thresholds on the decision function used to compute
            fpr and tpr. `thresholds[0]` represents no instances being predicted
            and is arbitrarily set to `max(y_score) + 1`.
    """
    # print(1)
    # y_true = y_true.astype(int)
    fpr, tpr, thresholds = skm.roc_curve(y_true, y_pred, pos_label=pos_label)
    return fpr, tpr, thresholds


def get_precision_recall_thr(y_true, y_pred, pos_label=1):
    precision, recall, thresholds = skm.precision_recall_curve(
        y_true, y_pred, pos_label=pos_label
    )
    return precision, recall, thresholds

def compute_auroc(fpr, tpr):
    return skm.auc(fpr, tpr)

def compute_fpr_tpr_thr_given_tpr_level(fpr, tpr, thresholds, tpr_level):
    if all(tpr < tpr_level):
        raise ValueError(f"No threshold allows for TPR at least {tpr_level}.")
    # elif all(tpr >= tpr_level):
    #     # All thresholds allow TPR >= tpr level, so find lowest possible FPR
    #     idx = np.argmin(fpr)
    else:
        idxs = [i for i, x in enumerate(tpr) if x >= tpr_level]
        idx = min(idxs)
    return fpr[idx], tpr[idx], thresholds[idx]

def compute_fpr_tpr(fpr, tpr, thresholds):
    idx = min(i for i, x in enumerate(tpr) if x >= 0.95)
    return fpr[idx], tpr[idx], thresholds[idx]

def compute_auc(pred_val, gt_val):
    gt_val = np.asarray(gt_val)
    pred_val = np.asarray(pred_val)
    fpr, tpr, thresholds = get_fpr_tpr_thr(gt_val, pred_val, pos_label=1)
    fpr_at_90tpr, _, _ = compute_fpr_tpr_thr_given_tpr_level(fpr, tpr, thresholds, tpr_level=0.9)
    precision, recall, thresholds = get_precision_recall_thr(gt_val, pred_val, pos_label=1)
    auc_roc = compute_auroc(fpr, tpr)
    auc = {"auc-ROC": auc_roc * 100, "fprat90tpr": fpr_at_90tpr * 100}
    return auc

def convert_to_numpy(s1_weights, s2_weights, C):
        if torch.is_tensor(s1_weights):
            s1_weights = s1_weights.to('cpu').numpy()
            s2_weights = s2_weights.to('cpu').numpy()
        if torch.is_tensor(C):
            C = C.to('cpu').numpy()
        return s1_weights, s2_weights, C