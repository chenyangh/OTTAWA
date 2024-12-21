from models import *
import torch
import numpy as np
import ot
from my_ot_algorithm import *
from utils import *
from models import *
import math


def for_analysis(src_rep, mt_rep, **kwargs):
    config = {
                    'epsilon': 0.02,
                    'direct_distance': True,
                    'dist': 'cosine',
                    'null_weight': 1.0,
                    'lbd': 0.5 
                }  
    
    error_type = kwargs['error_type']

    lbd = config['lbd']
    mt_rep = torch.stack(mt_rep, dim=0).cuda()
    src_rep = torch.stack(src_rep, dim=0).cuda()
    
    if kwargs['avg_emb'] is not None:
        avg_emb = kwargs['avg_emb']
    else:
        avg_emb = None
        
    val_method = 'min_max' #   min_max median
    src_rep = src_rep.to(torch.float64)
    mt_rep = mt_rep.to(torch.float64)
    
    # avg_c = find_avg_vector_tri(src_rep, mt_rep).to(src_rep).detach()
    # src_rep = F.normalize(torch.cat([src_rep, avg_c], dim=0))
    # mt_rep = F.normalize(torch.cat([mt_rep, avg_c], dim=0))
    
    P_rev, C_rev = get_ot_align(src_rep, mt_rep, config=config, val_method=val_method)
    null_idx_rev = P_rev.shape[1] - 1
    tgt_src_align = P_rev.argmax(-1)
    rev_align = set([(int(j), i)for i, j in enumerate(tgt_src_align) if j != null_idx_rev])
    
    P_fwd, C_fwd = get_ot_align(mt_rep, src_rep, config=config, val_method=val_method)
    null_idx_fwd = P_fwd.shape[1] - 1
    src_tgt_align = P_fwd.argmax(-1)
    fwd_align = set([(i, int(j)) for i, j in enumerate(src_tgt_align) if j != null_idx_fwd])
    

    or_align = fwd_align.union(rev_align)
    src_or_aligned_count = len(set([i[0] for i in or_align]))
    tgt_or_aligned_count = len(set([i[1] for i in or_align]))
    
    and_align = fwd_align.intersection(rev_align)
    src_and_aligned_count = len(set([i[0] for i in and_align]))
    tgt_and_aligned_count = len(set([i[1] for i in and_align]))
    
    confident_rev = P_rev[:, -1].sum()
    confident_fwd = P_fwd[:, -1].sum()
    
    hall_score = (len(mt_rep) - tgt_and_aligned_count)  / len(mt_rep)  +  confident_rev 
    
    omi_score = (len(src_rep) - src_and_aligned_count)  / len(src_rep)  
    
    hall_word_level = P_rev[:, -1].reshape(-1)
    
    omi_word_level = P_fwd[:, -1].reshape(-1)
    
    return hall_score, omi_score, hall_word_level, omi_word_level, P_rev, P_fwd, C_rev, C_fwd 


def ot_align_fwd_rev(src_rep, mt_rep, **kwargs):
    config = {
                    'epsilon': 0.05,
                    'direct_distance': True,
                    'dist': 'cosine',
                    'null_weight': 1.0,
                    'lbd': 0.5 
                }  
    
    error_type = kwargs['error_type']

    mt_rep = torch.stack(mt_rep, dim=0).cuda()
    src_rep = torch.stack(src_rep, dim=0).cuda()
    
    val_method = 'min_max' #   min_max median
    src_rep = src_rep.to(torch.float64)
    mt_rep = mt_rep.to(torch.float64)
    
    
    P_rev, _ = get_ot_align(src_rep, mt_rep, config=config, val_method=val_method)
    null_idx_rev = P_rev.shape[1] - 1
    tgt_src_align = P_rev.argmax(-1)
    rev_align = set([(int(j), i)for i, j in enumerate(tgt_src_align) if j != null_idx_rev])
    
    P_fwd, _ = get_ot_align(mt_rep, src_rep, config=config, val_method=val_method)
    null_idx_fwd = P_fwd.shape[1] - 1
    src_tgt_align = P_fwd.argmax(-1)
    fwd_align = set([(i, int(j)) for i, j in enumerate(src_tgt_align) if j != null_idx_fwd])
    

    or_align = fwd_align.union(rev_align)
    src_or_aligned_count = len(set([i[0] for i in or_align]))
    tgt_or_aligned_count = len(set([i[1] for i in or_align]))
    
    and_align = fwd_align.intersection(rev_align)
    src_and_aligned_count = len(set([i[0] for i in and_align]))
    tgt_and_aligned_count = len(set([i[1] for i in and_align]))
    
    confident_rev = P_rev[:, -1].sum()
    
    if error_type == 'hall':
        score =  (len(mt_rep) - tgt_and_aligned_count)  / len(mt_rep)  +  confident_rev   
    else:
        score = (len(src_rep) - src_and_aligned_count)  / len(src_rep)
        
    return score




def partial_ot_hall_word(src_rep, mt_rep, **kwargs):
    config = {
                    'epsilon': 0.02,
                    'direct_distance': True,
                    'dist': 'cosine',
                    'null_weight': 1.0,
                }  
    
        
    mt_rep = torch.stack(mt_rep, dim=0)
    src_rep = torch.stack(src_rep, dim=0)
    
    C = compute_distance_matrix_cosine(mt_rep, src_rep, 0.0).to(src_rep)    
    s1_weights, s2_weights = compute_weights_uniform(mt_rep, src_rep)
    s1_weights = s1_weights / s1_weights.sum()
    s2_weights = s2_weights / s2_weights.sum()
    
    C = min_max_scaling(C)
    s1_weights, s2_weights, C = convert_to_numpy(s1_weights, s2_weights, C)
    P = ot.emd(s1_weights, s2_weights, C)
    forward_best = P.argmax(-1)
    forward_align = set([(i, forward_best[i]) for i in range(len(forward_best))])
    
    reverse_best = P.argmax(0)
    reverse_align = set([(reverse_best[i], i) for i in range(len(reverse_best))])
    
    intersect = forward_align.intersection(reverse_align)
    
    interest_forward_missing =   (len(mt_rep) - len(intersect) ) / len(mt_rep)
    
    return interest_forward_missing


def other_alignment_methods(src_rep, mt_rep, **kwargs):
       
    mt_rep = torch.stack(mt_rep, dim=0)
    src_rep = torch.stack(src_rep, dim=0)
    method = kwargs['align_method']
    error = kwargs["error_type"]

    C = compute_distance_matrix_cosine(src_rep, mt_rep, 0.0).to(src_rep)    
    
    C_sim = 1 - C.clone()
    
    C = min_max_scaling(C)
    
    def get_threshold_map(C_sim, percentile=95):
        # Normalize similarity matrix row-wise and column-wise
        normalize_C_fwd = C_sim / C_sim.sum(-1, keepdim=True)
        normalize_C_rev = C_sim / C_sim.sum(0, keepdim=True)

        # Calculate normalized entropy for forward and reverse normalization
        entropy_C_fwd = -torch.sum(normalize_C_fwd * torch.log(normalize_C_fwd + 1e-10), dim=-1) / torch.log(torch.tensor(C_sim.size(-1), dtype=torch.float))
        entropy_C_rev = -torch.sum(normalize_C_rev * torch.log(normalize_C_rev + 1e-10), dim=0) / torch.log(torch.tensor(C_sim.size(0), dtype=torch.float))

        # Get the minimum entropy values between corresponding elements of entropy_C_fwd and entropy_C_rev
        min_entropy_values = torch.min(entropy_C_fwd.unsqueeze(-1).expand_as(C_sim), entropy_C_rev.expand_as(C_sim))

        # Calculate the threshold using the specified percentile on the min entropy values
        thres = torch.quantile(min_entropy_values, percentile / 100.0)

        # Initialize threshold map with ones
        threshold_map = torch.ones_like(C_sim)

        # Apply the threshold
        # If the min entropy of the pair (i, j) is greater than the threshold, set to 0
        threshold_map[min_entropy_values > thres] = 0

        return threshold_map

            
    if method == 'ot':
        s1_weights, s2_weights = compute_weights_uniform(src_rep, mt_rep)
        s1_weights = s1_weights / s1_weights.sum()
        s2_weights = s2_weights / s2_weights.sum()
        
        s1_weights, s2_weights, C = convert_to_numpy(s1_weights, s2_weights, C)
        # P = ot.emd(s1_weights, s2_weights, C)
        P = ot.sinkhorn(s1_weights, s2_weights, C, 0.1)
        
    elif method == 'pmi':
        def pmi_matrix(out_src, out_tgt):
            out_src.div_(torch.norm(out_src, dim=-1).unsqueeze(-1))
            out_tgt.div_(torch.norm(out_tgt, dim=-1).unsqueeze(-1))

            sim = torch.matmul(out_src, out_tgt.transpose(-1, -2))
            
            sim = torch.softmax(sim.view(-1), dim=0).view(sim.size())

            probs_src = torch.sum(sim, dim=1)
            probs_tgt = torch.sum(sim, dim=0)

            repeat_probs_src = probs_src.unsqueeze(1).expand(-1, sim.size(-1))
            repeat_probs_tgt = probs_tgt.repeat(sim.size(0), 1)
            scores = torch.log(sim) - torch.log(repeat_probs_tgt) - torch.log(repeat_probs_src)

            scores = (scores - scores.min()) / (scores.max() - scores.min())

            return scores

        out_src = src_rep
        out_tgt = mt_rep

        out_src.div_(torch.norm(out_src, dim=-1).unsqueeze(-1))
        out_tgt.div_(torch.norm(out_tgt, dim=-1).unsqueeze(-1))

        P = pmi_matrix(out_src, out_tgt)
    elif method == 'argmax':
        P = C
        
    elif method == 'itermax':     
        def __iter_max(sim_matrix: np.ndarray, max_count: int = 2) -> np.ndarray:
            alpha_ratio = 0.8
            m, n = sim_matrix.shape
            forward = np.eye(n)[sim_matrix.argmax(axis=1)]  # m x n
            backward = np.eye(m)[sim_matrix.argmax(axis=0)]  # n x m
            inter = forward * backward.transpose()

            if min(m, n) <= 2:
                return inter

            new_inter = np.zeros((m, n))
            count = 1
            while count < max_count:
                mask_x = 1.0 - np.tile(inter.sum(1)[:, np.newaxis], (1, n)).clip(0.0, 1.0)
                mask_y = 1.0 - np.tile(inter.sum(0)[np.newaxis, :], (m, 1)).clip(0.0, 1.0)
                mask = ((alpha_ratio * mask_x) + (alpha_ratio * mask_y)).clip(0.0, 1.0)
                mask_zeros = 1.0 - ((1.0 - mask_x) * (1.0 - mask_y))
                if mask_x.sum() < 1.0 or mask_y.sum() < 1.0:
                    mask *= 0.0
                    mask_zeros *= 0.0

                new_sim = sim_matrix * mask
                fwd = np.eye(n)[new_sim.argmax(axis=1)] * mask_zeros
                bac = np.eye(m)[new_sim.argmax(axis=0)].transpose() * mask_zeros
                new_inter = fwd * bac

                if np.array_equal(inter + new_inter, inter):
                    break
                inter = inter + new_inter
                count += 1
            return inter
        binary_map = __iter_max(C.cpu().numpy(), max_count=2)
        thre_map = get_threshold_map(C_sim, percentile=50)
        binary_map = binary_map * thre_map.numpy()
        
    elif method == 'pot':
        s1_weights, s2_weights = compute_weights_uniform(src_rep, mt_rep)
        s1_weights = s1_weights / s1_weights.sum()
        s2_weights = s2_weights / s2_weights.sum()
        s1_weights, s2_weights, C = convert_to_numpy(s1_weights, s2_weights, C)
        
        m = 0.5
        threshold = 1 / max(len(src_rep), len(mt_rep)) * 0.025  # 0.1 0.05 0.025
          
        P = ot.partial.entropic_partial_wasserstein(s1_weights, s2_weights, C, reg=0.1, m=m)
        
        binary_map = (P > threshold).astype(int)
        
    if method in ['ot', 'pmi', 'argmax']:
        forward_best = P.argmax(-1)
        
        forward_align = set([(i, int(forward_best[i])) for i in range(len(forward_best))])
        reverse_best = P.argmax(0)
        reverse_align = set([(int(reverse_best[i]), i) for i in range(len(reverse_best))])
        intersect = forward_align.intersection(reverse_align)
        
        num_src_aligned = set([i[0] for i in intersect])
        num_tgt_aligned = set([i[1] for i in intersect])
        
        if error == 'hall':
            return (len(mt_rep) - len(num_tgt_aligned) ) / len(mt_rep)
        elif error == 'omi':
            return (len(src_rep) - len(num_src_aligned)) / len(src_rep)
        
    if method in ['itermax', 'pot']:
        src_alined = (binary_map.sum(1) > 0).sum()
        tgt_aligned = (binary_map.sum(0) > 0).sum()
        
        if error == 'hall':
            return (len(mt_rep) - tgt_aligned) / len(mt_rep)

        elif error == 'omi':
            return (len(src_rep) - src_alined) / len(src_rep)

            

def get_ot_align(src_rep, mt_rep, config, val_method='min', avg_emb=None):
    # m=m, epsilon=epsilon, numItermax=numItermax, stopThr=stopThr, 
    """Default configuration for OT alignment
        m = 1.0
        epsilon = 0.1
        numItermax = 2000
        stopThr = 1e-6
        dist = 'l2
    """
    direct_distance = config.get('direct_distance', True)
    m = config.get('m', 1.0)
    epsilon = config.get('epsilon', 0.1)
    numItermax = config.get('numItermax', 2000)
    stopThr = config.get('stopThr', 1e-6)
    dist = config.get('dist', 'l2')
    null_weight = config.get('null_weight', 1.0)
    
    
    if type(src_rep) is list:
        src_rep = torch.stack(src_rep, dim=0)
        mt_rep = torch.stack(mt_rep, dim=0)
    
    # C = compute_distance_matrix_l2(mt_rep, src_rep, 0.0).to(src_rep)
    if dist == 'l2':
        C = compute_distance_matrix_l2(mt_rep, src_rep, 0.0).to(src_rep)
    elif dist == 'cosine':
        C = compute_distance_matrix_cosine(mt_rep, src_rep, 0.0).to(src_rep)
    else:
        raise ValueError(f"Unknown distance metric: {dist}")

    s1_weights, s2_weights = compute_weights_uniform(mt_rep, src_rep)
    s1_weights = s1_weights.to(C)
    s2_weights = s2_weights.to(C)
    
    if direct_distance:
        if val_method == 'min':
            assert dist != 'cosine'
            min_C = get_equal_n_min_dist(src_rep, mt_rep)
            C_val = min_C
        elif val_method == 'median':
            median = torch.quantile(C.view(-1), q=0.5)
            C_val = median
        elif val_method == 'avg':
            avg_C = C.mean()
            C_val = avg_C
        elif val_method == 'mid':
            h = C.max()
            l = C.min()
            mid_C = (h + l) / 2
            C_val = mid_C
        elif val_method == 'min_max':
            src_rep = F.normalize(src_rep)
            mt_rep = F.normalize(mt_rep)
            median = torch.quantile(C.view(-1), q=0.5)
            # C_min = find_avg_vector_exact(src_rep)
            C_min = find_avg_vector_exact(mt_rep)
            C_val = max(C_min, median)

        C = torch.cat([C, torch.ones(C.shape[0], 1).to(C) * C_val], dim=1)
        s2_weights = torch.cat([s2_weights, torch.tensor([1.0]).to(C)], dim=0)

    
    s1_weights = s1_weights / s1_weights.sum()
    s2_weights = s2_weights / s2_weights[:-1].sum()
    s2_weights[-1] = null_weight
    
    C = min_max_scaling(C)
    s1_weights, s2_weights, C = convert_to_numpy(s1_weights, s2_weights, C)

    m = s1_weights.sum()
    P = ot.partial.entropic_partial_wasserstein(s1_weights, s2_weights, C, reg=epsilon, m=m, stopThr=stopThr, numItermax=numItermax)
    
    return P, C

def create_top_p_mask(scores, p):
    cutoff_index = math.ceil(len(scores) * p)
    sorted_indices = np.argsort(scores)[::-1]
    mask = np.full(len(scores), False)
    mask[sorted_indices[:cutoff_index]] = True
    return mask

def get_log_weight(log_probabilities, type="none"):
    if type == 'norm':
        # Find the minimum log probability
        min_log_prob = np.min(log_probabilities)
        # Calculate the offset (make it slightly larger than the absolute value of min_log_prob)
        offset = abs(min_log_prob) + 1  # Adding a small value like 0.01 for a safety margin
        positive_weights = log_probabilities + offset
        return positive_weights
    elif type == 'neg':
        return - log_probabilities
    else:
        return None

def find_avg_vector(vectors, vectors2=None):
    def objective_function_uniform_l2(x, vectors):
        distances = torch.norm(vectors - x, dim=1)
        return torch.var(distances)
    
    def objective_function_uniform_cos(x, vectors):
        distances = 1 - torch.matmul(F.normalize(vectors), F.normalize(x).T) 
        return torch.var(distances)
        
    def objective_function_sum_l2(x, vectors, ):
        distances = torch.norm(vectors - x, dim=1)
        return distances.mean()
    
    def objective_function_sum_cos(x, vectors, ):
        distances = 1 - torch.matmul(F.normalize(vectors), F.normalize(x).T)  #
        return distances.mean() # distances.norm()
    
    if type(vectors) is list:
        vectors = torch.stack(vectors, dim=0)
        if vectors2 is not None:
            vectors2 = torch.stack(vectors2, dim=0)
    
    if vectors2 is not None:
        combined_vectors = torch.cat([vectors, vectors2], dim=0)
    else:
        combined_vectors = vectors
        

    x = torch.mean(combined_vectors.detach(), dim=0, keepdim=True).clone().detach().requires_grad_(True)
    
    optimizer = torch.optim.Adam([x], lr=0.02)
    
    over_all_loss = float('inf')
    for _ in range(200):
        previous_loss = float('inf')
        for _ in range(3000):  # Number of iterations
            optimizer.zero_grad()
            loss = objective_function_sum_cos(x, combined_vectors.detach())
            loss.backward()
            optimizer.step()
            if torch.abs(previous_loss - loss) < 1e-7:
                break
            previous_loss = loss
            
        optimizer = torch.optim.Adam([x], lr=0.01)
        previous_loss = float('inf')
        for _ in range(3000):  # Number of iterations
            optimizer.zero_grad()
            loss = objective_function_uniform_cos(x, combined_vectors.detach())
            loss.backward()
            optimizer.step()
            if torch.abs(previous_loss - loss) < 1e-7:
                break
            previous_loss = loss
        
        if torch.abs(over_all_loss - loss) < 1e-7:
            break
        over_all_loss = loss
        
    return x.detach().squeeze(0)


def find_avg_vector_exact(vectors, vectors2=None):
    if vectors2 is not None:
        combined_vectors = torch.cat([vectors, vectors2], dim=0)
    else:
        combined_vectors = vectors
    v = F.normalize(combined_vectors)
    
    v_jk = (v[0].unsqueeze(0) - v[1:]) @ v.T  # double check
    
    I = torch.eye(v.shape[0]).to(v)
    
    # e = torch.rand(v.shape[0]).to(v)
    
    def get_dist(v, v_jk, i):
        # a = (I  - v_jk.pinverse() @ v_jk) @ e
        
        A = (I  - v_jk.pinverse() @ v_jk)
        
        a = A[:, i]
        
        v_c = a.unsqueeze(0) @ v
        
        v_c = F.normalize(v_c)
        
        cos_dist = 1 - (torch.matmul(v[0], v_c.T) + 1) / 2
        
        return cos_dist
    
    N = v.shape[0]

    cos_dist = min([get_dist(v, v_jk, i) for i in range(N)])
    
    return cos_dist


def find_avg_vector_tri(x, y):    
    def compute_pair_wise_cosine_distance(tensor_a, tensor_b):
        return 1 - (torch.matmul(F.normalize(tensor_a), F.normalize(tensor_b).T) + 1) / 2
    
    def variance_objective(c, x, y, lambda_weight):
        combined_x_y = torch.cat([x, y], dim=0)
        xy_distances = compute_pair_wise_cosine_distance(x, y)
        mask = torch.ones_like(xy_distances, dtype=torch.bool)
        mask.fill_diagonal_(0)
        xy_distances = xy_distances[mask.bool()].reshape(-1)
        
        c_to_xy_distances = compute_pair_wise_cosine_distance(c, combined_x_y).reshape(-1)
        
        # c_to_y_distances = compute_pair_wise_cosine_distance(c, x).reshape(-1)
        
        # all_distancs = torch.cat([xy_distances, c_to_xy_distances], dim=0)

        pair_wise_distances = (c_to_xy_distances.unsqueeze(1) - xy_distances.unsqueeze(0)).reshape(-1)
        
        return pair_wise_distances.abs().sum() 
        
        

    # Parameters
    alpha = 0.5
    lambda_weight = 0.5
    learning_rate = 0.05
    max_iterations = 3000
    convergence_threshold = 1e-7

    # Initialize c
    c = torch.mean(torch.cat([x.detach(), y.detach()], dim=0), dim=0).clone().reshape(1, -1).requires_grad_(True).to(x)

    # Optimizer
    optimizer = torch.optim.Adam([c], lr=learning_rate)

    # Iterative Process    
    previous_loss_var = float('inf')
    for _idx in range(max_iterations):
        optimizer.zero_grad()
        loss_variance = variance_objective(c, x.detach(), y.detach(), lambda_weight)
        loss_variance.backward()
        optimizer.step()

        # Early Stopping for Step 2
        if torch.abs(previous_loss_var - loss_variance) < convergence_threshold:
            break
        previous_loss_var = loss_variance

    # print(iteration)
    return c