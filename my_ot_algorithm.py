# word_level_emb = get_word_embeddings("This is an example sentence , which is meant to be encoded . Subwords are not a problem for this model .", model, model.tokenizer)
import torch
from align_utils import *
import ot
from collections import Counter

# epsilon = 0.1
numItermax = 2000
stopThr = 1e-6

def normalize_weights(s1_weights, s2_weights):
    s1_weights = s1_weights / s1_weights.sum() 
    s2_weights = s2_weights / s2_weights.sum()
    return s1_weights, s2_weights

def convert_to_numpy(s1_weights, s2_weights, C):
        if torch.is_tensor(s1_weights):
            s1_weights = s1_weights.to('cpu').numpy()
            s2_weights = s2_weights.to('cpu').numpy()
        if torch.is_tensor(C):
            C = C.to('cpu').numpy()
        return s1_weights, s2_weights, C


def get_equal_n_min_dist(src_rep, ref_rep):
    src_rep = F.normalize(src_rep, dim=-1)
    ref_rep = F.normalize(ref_rep, dim=-1)
    x = torch.cat([src_rep, ref_rep], dim=0)
    b = (x.norm(dim=1)[0].unsqueeze(0) ** 2 - x.norm(dim=1)[1:] ** 2) / 2
    A = x[0].unsqueeze(0) - x[1:]
    A_pinv = A.pinverse()
    d = (A_pinv @ ( A @ x[0] - b)).norm()
    
    # d = (x[0] - A_pinv @ b).norm()
    return d

    
def get_ot_map(src_rep, mt_rep, config=None):
    """
        config: {
            'method': 'emd', 'sink', 'p_sink', 'p_sink_fwd', 'p_sink_bwd', 'unk_sink_fwd', 'unk_sink_bwd'
            'epsilon': 0.1,
            'relax_ratio': 1.5,
            'distance_metric': 'cosine', 'l2'
        }
    """
    if type(src_rep) == list:
        src_rep = torch.stack(src_rep, dim=0).cuda()
        mt_rep = torch.stack(mt_rep, dim=0).cuda()
    
    distance_metric = config['distance_metric'] if 'distance_metric' in config else 'cosine'
    config['relax_ratio'] = config['relax_ratio'] if 'relax_ratio' in config else 1.5
    
    
    distance_metric =  'cosine'
    if distance_metric == 'cosine':
        C = compute_distance_matrix_cosine(src_rep, mt_rep, 0.0)
    elif distance_metric == 'l2':
        C = compute_distance_matrix_l2(src_rep, mt_rep, 0.0)
    else:
        raise Exception('Unknown distance metric')
    
    s1_weights, s2_weights = compute_weights_uniform(src_rep, mt_rep)
    
    if config['method'] == 'emd':
        get_equal_n_min_dist(src_rep, mt_rep)
        s1_weights, s2_weights = normalize_weights(s1_weights, s2_weights)
        P = ot.emd(s1_weights, s2_weights, C)
    elif config['method'] == 'sink':
        s1_weights, s2_weights = normalize_weights(s1_weights, s2_weights)
        P = ot.partial.entropic_partial_wasserstein(s1_weights, s2_weights, C, m=0.999, reg=config['epsilon'], numItermax=numItermax, stopThr=stopThr)       
        # P = ot.sinkhorn(s1_weights, s2_weights, C, reg=config['epsilon'], numItermax=numItermax, stopThr=stopThr)       

    elif config['method'] == 'p_sink':
        s1_weights, s2_weights = normalize_weights(s1_weights, s2_weights)
        s1_weights *= config['relax_ratio']
        s2_weights *= config['relax_ratio']
        P = ot.partial.entropic_partial_wasserstein(s1_weights, s2_weights, C, m=0.999, reg=config['epsilon'], numItermax=numItermax, stopThr=stopThr)
    elif config['method'] == 'p_sink_f2p':
        s1_weights, s2_weights = normalize_weights(s1_weights, s2_weights)
        s2_weights *= config['relax_ratio']
        P = ot.partial.entropic_partial_wasserstein(s1_weights, s2_weights, C, m=0.999, reg=config['epsilon'], numItermax=numItermax, stopThr=stopThr)
    elif config['method'] == 'p_sink_p2f':
        s1_weights, s2_weights = normalize_weights(s1_weights, s2_weights)
        s1_weights *=  config['relax_ratio']
        P = ot.partial.entropic_partial_wasserstein(s1_weights, s2_weights, C, m=0.999, reg=config['epsilon'], numItermax=numItermax, stopThr=stopThr)
    elif config['method'] == 'p_sink_f2f_fwd':
        # avg_C = C.mean()
        # avg_C = get_equal_n_min_dist(src_rep, mt_rep)
        avg_C = torch.quantile(C.view(-1), q=0.5)
        C = torch.cat([C, torch.ones(C.shape[0], 1).to(C) * avg_C], dim=1)
        s2_weights = torch.cat([s2_weights, torch.tensor([1.0]).to(C)], dim=0)
        s1_weights, s2_weights = normalize_weights(s1_weights, s2_weights)
        P = ot.partial.entropic_partial_wasserstein(s1_weights, s2_weights, C, m=0.999, reg=config['epsilon'], numItermax=numItermax, stopThr=stopThr)
        P = torch.relu(P[:, :-1] - P[:, -1].unsqueeze(1))
    elif config['method'] == 'p_sink_f2f_rev':
        # avg_C = C.mean()
        avg_C = torch.quantile(C.view(-1), q=0.5)
        # avg_C = get_equal_n_min_dist(src_rep, mt_rep)
        C = torch.cat([C, torch.ones(1, C.shape[1]).to(C) * avg_C], dim=0)
        s1_weights = torch.cat([s1_weights, torch.tensor([1.0]).to(C)], dim=0)
        s1_weights, s2_weights = normalize_weights(s1_weights, s2_weights)
        P = ot.partial.entropic_partial_wasserstein(s1_weights, s2_weights, C, m=0.999, reg=config['epsilon'], numItermax=numItermax, stopThr=stopThr)
        P = torch.relu(P[:-1, :] - P[-1, :].unsqueeze(0))
    elif config['method'] == 'p_sink_p2f_fwd':
        # avg_C = C.mean()
        avg_C = torch.quantile(C.view(-1), q=0.5)
        # avg_C = get_equal_n_min_dist(src_rep, mt_rep)
        C = torch.cat([C, torch.ones(C.shape[0], 1).to(C) * avg_C], dim=1)
        s2_weights = torch.cat([s2_weights, torch.tensor([1.0]).to(C)], dim=0)
        s1_weights, s2_weights = normalize_weights(s1_weights, s2_weights)
        s1_weights *= config['relax_ratio']
        P = ot.partial.entropic_partial_wasserstein(s1_weights, s2_weights, C, m=0.999, reg=config['epsilon'], numItermax=numItermax, stopThr=stopThr)
        P = torch.relu(P[:, :-1] - P[:, -1].unsqueeze(1))
    elif config['method'] == 'p_sink_f2p_rev':
        # avg_C = C.mean()
        avg_C = torch.quantile(C.view(-1), q=0.5)
        # avg_C = get_equal_n_min_dist(src_rep, mt_rep)
        C = torch.cat([C, torch.ones(1, C.shape[1]).to(C) * avg_C], dim=0)
        s1_weights, s2_weights = normalize_weights(s1_weights, s2_weights)
        s1_weights = torch.cat([s1_weights, torch.tensor([1.0]).to(C)], dim=0)
        # s2_weights *= config['relax_ratio']
        P = ot.partial.entropic_partial_wasserstein(s1_weights, s2_weights, C, m=0.999, reg=config['epsilon'], numItermax=numItermax, stopThr=stopThr)
        P = torch.relu(P[:-1, :] - P[-1, :].unsqueeze(0))
    elif config['method'] == 'p_sink_f2p_fwd':
        avg_C = torch.quantile(C.view(-1), q=0.5)
        # avg_C = get_equal_n_min_dist(src_rep, mt_rep)
        C = torch.cat([C, torch.ones(C.shape[0], 1).to(C) * avg_C], dim=1)
        s1_weights, s2_weights = normalize_weights(s1_weights, s2_weights)
        s2_weights = torch.cat([s2_weights, torch.tensor([1.0]).to(C)], dim=0)
        P = ot.partial.entropic_partial_wasserstein(s1_weights, s2_weights, C, m=0.999, reg=config['epsilon'], numItermax=numItermax, stopThr=stopThr)
        P = torch.relu(P[:, :-1] - P[:, -1].unsqueeze(1))
    elif config['method'] == 'p_sink_p2f_rev':
        avg_C = torch.quantile(C.view(-1), q=0.5)
        C = torch.cat([C, torch.ones(1, C.shape[1]).to(C) * avg_C], dim=0)
        s1_weights, s2_weights = normalize_weights(s1_weights, s2_weights)
        s1_weights = torch.cat([s1_weights, torch.tensor([1.0]).to(C)], dim=0)
        P = ot.partial.entropic_partial_wasserstein(s1_weights, s2_weights, C, m=0.999, reg=config['epsilon'], numItermax=numItermax, stopThr=stopThr)
        P = torch.relu(P[:-1, :] - P[-1, :].unsqueeze(0))    
    else:
        raise Exception('Unknown method')
    
    return_val = {
        "map": P,
        "config": config,
        "max_val_x": s1_weights,
        "max_val_y": s2_weights,        
    }
    return return_val


def ot_mbr(P_dict, C):
    num_src, num_mt = next(iter(P_dict.items()))[1].shape
    # NOTE simple majority vote
    fwd_alignments = {}
    for i in range(num_src):
        fwd_alignments[i] = {"hard_vote": []}
        for method, P in P_dict.items():
            
            P = P / P.sum()
            j = -1 if P[i].sum() == 0 else P[i].argmax()
            fwd_alignments[i]["hard_vote"].append(j)
        
        # counting votes    
        cout = Counter(fwd_alignments[i]["hard_vote"])
        majority_vote = cout.most_common(1)[0][0]
        # if majority_vote == -1:
        #     print('found NULL alignment')
            
        fwd_alignments[i]["majority_vote"] = majority_vote
        
    rev_alignments = {}
    for j in range(num_mt):
        rev_alignments[j] = {"hard_vote": []}
        for method, P in P_dict.items():
            P = P / P.sum()
            i = -1 if P[:, j].sum() == 0 else P[:, j].argmax()
            rev_alignments[j]["hard_vote"].append(i)
        
        # if -1 in rev_alignments[j]["hard_vote"]:
        #     print('found NULL alignment')
        # counting votes
        cout = Counter(rev_alignments[j]["hard_vote"])
        majority_vote = cout.most_common(1)[0][0]
        # if majority_vote == -1:
        #     print('found NULL alignment')

        rev_alignments[j]["majority_vote"] = majority_vote
        
    # convert to alignment string
    fwd_alignment_str = ' '.join([f"{i+1}-{fwd_alignments[i]['majority_vote']+1}" for i in range(num_src) if "majority_vote" in fwd_alignments[i]] )
    
    rev_alignment_str = ' '.join([f"{rev_alignments[j]['majority_vote']+1}-{j+1}" for j in range(num_mt) if "majority_vote" in rev_alignments[j]])
    
    return fwd_alignment_str, rev_alignment_str
    
    

      
            

        

        
        

      
        
        
        



