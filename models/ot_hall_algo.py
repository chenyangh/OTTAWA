# import torch
# import ot
# import math


# from models.ot_hall_utils  import *

# # def get_ot_align(src_rep, mt_rep, config):
# #     # m=m, epsilon=epsilon, numItermax=numItermax, stopThr=stopThr, 
    
# #     """Default configuration for OT alignment
# #         m = 1.0
# #         epsilon = 0.1
# #         numItermax = 2000
# #         stopThr = 1e-6
# #         dist = 'l2
# #     """
    
# #     direct_distance = config.get('direct_distance', False)
# #     m = config.get('m', 1.0)
# #     epsilon = config.get('epsilon', 0.1)
# #     numItermax = config.get('numItermax', 2000)
# #     stopThr = config.get('stopThr', 1e-6)
# #     dist = config.get('dist', 'l2')
# #     null_weight = config.get('null_weight', 1.0)
    
# #     def convert_to_numpy(s1_weights, s2_weights, C):
# #         if torch.is_tensor(s1_weights):
# #             s1_weights = s1_weights.to('cpu').numpy()
# #             s2_weights = s2_weights.to('cpu').numpy()
# #         if torch.is_tensor(C):
# #             C = C.to('cpu').numpy()
# #         return s1_weights, s2_weights, C
    
# #     # C = compute_distance_matrix_l2(mt_rep, src_rep, 0.0).to(src_rep)
# #     if dist == 'l2':
# #         C = compute_distance_matrix_l2(mt_rep, src_rep, 0.0).to(src_rep)
# #     elif dist == 'cosine':
# #         C = compute_distance_matrix_cosine(mt_rep, src_rep, 0.0).to(src_rep)
# #     else:
# #         raise ValueError(f"Unknown distance metric: {dist}")

    
# #     s1_weights, s2_weights = compute_weights_uniform(mt_rep, src_rep)
# #     s1_weights = s1_weights.to(C)
# #     s2_weights = s2_weights.to(C)
    
# #     if direct_distance:
# #         avg_C = C.mean()
# #         # src_rep = F.normalize(src_rep)
# #         # mt_rep = F.normalize(mt_rep)
# #         # avg_C = get_equal_n_min_dist(src_rep, mt_rep)
# #         C = torch.cat([C, torch.ones(C.shape[0], 1).to(C) * avg_C], dim=1)
# #         s2_weights = torch.cat([s2_weights, torch.tensor([1.0]).to(C)], dim=0)

# #     # s1_weights, s2_weights = compute_weights_norm(mt_rep, src_rep)
# #     # s2_weights[-1] = 2.0
# #     s1_weights = s1_weights / s1_weights.sum()
# #     s2_weights = s2_weights / s2_weights[:-1].sum()
# #     s2_weights[-1] = null_weight
# #     # s2_weights *= 1.5
    
# #     # s2_weights[:-1].sum() * 0.5
# #     C = min_max_scaling(C)
# #     s1_weights, s2_weights, C = convert_to_numpy(s1_weights, s2_weights, C)

# #     m = np.min((np.sum(s1_weights), np.sum(s2_weights))) * m
# #     P = ot.partial.entropic_partial_wasserstein(s1_weights, s2_weights, C, reg=epsilon, m=m, stopThr=stopThr, numItermax=numItermax)
# #     # P = min_max_scaling(P)
# #     # P = ot.unbalanced.sinkhorn_stabilized_unbalanced(s1_weights, s2_weights, C, reg=epsilon, reg_m=(0.1, 1), stopThr=stopThr, numItermax=numItermax)
# #     # P = min_max_scaling(P)
    
# #     # P = ot.emd(s1_weights / s1_weights.sum(), s2_weights / s2_weights.sum(), C)    
# #     return P, C

# def create_top_p_mask(scores, p):
#     cutoff_index = math.ceil(len(scores) * p)
#     sorted_indices = np.argsort(scores)[::-1]
#     mask = np.full(len(scores), False)
#     mask[sorted_indices[:cutoff_index]] = True
#     return mask

# def get_log_weight(log_probabilities, type="none"):
#     if type == 'norm':
#         # Find the minimum log probability
#         min_log_prob = np.min(log_probabilities)
#         # Calculate the offset (make it slightly larger than the absolute value of min_log_prob)
#         offset = abs(min_log_prob) + 1  # Adding a small value like 0.01 for a safety margin
#         positive_weights = log_probabilities + offset
#         return positive_weights
#     elif type == 'neg':
#         return - log_probabilities
#     else:
#         return None


# # def get_equal_n_min_dist(src_rep, ref_rep):
# #     # src_rep = F.normalize(src_rep, dim=-1)
# #     # ref_rep = F.normalize(ref_rep, dim=-1)
# #     x = torch.cat([src_rep, ref_rep], dim=0)
# #     b = (x.norm(dim=1)[0].unsqueeze(0) ** 2 - x.norm(dim=1)[1:] ** 2) / 2
# #     A = x[0].unsqueeze(0) - x[1:]
# #     A_pinv = A.pinverse()
# #     d = (A_pinv @ ( A @ x[0] - b)).norm()
    
# #     # d = (x[0] - A_pinv @ b).norm()
# #     return d


