import torch
import numpy as np
import ot
from my_ot_algorithm import *
from utils import *

####################
#### PMI Align #####
####################

def generic_ot_align_test(out_src, out_tgt, sub2word_map_src, sub2word_map_tgt):
    
    return_val = get_ot_map(out_src, out_tgt, 
                      config={'method': 'sink',
                              'epsilon': 0.1,
                              'distance_metric': 'cosine',
                              # 'relax_ratio': 1.5
                              })
    P = return_val['map']
    
    argmax_srctgt = torch.argmax(P, dim=1)
    argmax_srctgt[P.sum(1) == 0] = -1
    align_words_srctgt = set()
    for i, j in enumerate(argmax_srctgt):
        if j == -1:
            continue
        align_words_srctgt.add(((sub2word_map_src[i], sub2word_map_tgt[j])))
                
    # P, _ = get_ot_map(out_src, out_tgt, 
    #                   config={'method': 'p_sink_f2f_rev',
    #                           'epsilon': 0.1,
    #                           'distance_metric': 'l2',
    #                           })

    argmax_tgtsrc = torch.argmax(P, dim=0)
    argmax_tgtsrc[P.sum(0) == 1] = -1
    align_words_tgtsrc = set()
    for i, j in enumerate(argmax_tgtsrc):
        if j == -1:
            continue
        align_words_tgtsrc.add((sub2word_map_src[j], sub2word_map_tgt[i]))

    align_words = align_words_srctgt.intersection(align_words_tgtsrc)
    # indices = sinkhorn(dot_prod)
    # align_words = set()
    # for p in indices:
    #   align_words.add( (sub2word_map_src[p[0]], sub2word_map_tgt[p[1]]) )

    alignStr = ""
    for p in align_words:
        alignStr += str(p[0]) + "-" + str(p[1]) + " "
    return alignStr

def get_top_P_pred(P, i=None, j=None, top_p=1, max_val=1.0):
    def normalize_line(a_map, max_val):
        return a_map / max_val
    if i is not None and j is None:
        P_i = P[i]
        P_i_norm = normalize_line(P_i, max_val)
        top_p_idx = get_top_p_indices(P_i_norm, top_p)
        return {int(i): float(P_i[i]) for i in top_p_idx}
    elif j is not None and i is None:
        P_j = P[:, j]
        P_j_norm = normalize_line(P_j, max_val)
        top_p_idx = get_top_p_indices(P_j_norm, top_p)
        return {int(j): float(P_j[j]) for j in top_p_idx}
    else:
        raise ValueError('i and j cannot be None at the same time')    



def ot_mbr_align(out_src, out_tgt, sub2word_map_src, sub2word_map_tgt):
    P_dict = ot_mbr_get_maps(out_src, out_tgt)
    src_len, tgt_len = out_src.size(0), out_tgt.size(0)
    
    # Consider src to tgt alignment    
    # assume fertility is 1
    fer_1_src_tgt_map = {}
    for src_idx in range(src_len):
        fer_1_src_tgt_map[src_idx] = {}
        for P_val in P_dict.values():
            if P_val['config']['method'] in ['rev']:
                continue
            P_map = P_val['map']
            max_val_x = P_val['max_val_x']
            top_p_align_idx = get_top_P_pred(P_map, i=src_idx, top_p=1.0, max_val=max_val_x[src_idx])
            fer_1_src_tgt_map[src_idx] = update_top_choices(fer_1_src_tgt_map[src_idx], top_p_align_idx)   
        fer_1_src_tgt_map[src_idx] = sum_sort_top_1(fer_1_src_tgt_map[src_idx], 1)
        
    fer_1_tgt_src_map = {}
    for tgt_idx in range(tgt_len):
        fer_1_tgt_src_map[tgt_idx] = {}
        for P_val in P_dict.values():
            if P_val['config']['method'] in ['fwd']:
                continue
            P_map = P_val['map']
            max_val_y = P_val['max_val_y']
            top_p_align_idx = get_top_P_pred(P_map, j=tgt_idx, top_p=1.0, max_val=max_val_y[tgt_idx])
            fer_1_tgt_src_map[tgt_idx] = update_top_choices(fer_1_tgt_src_map[tgt_idx], top_p_align_idx)
        fer_1_tgt_src_map[tgt_idx] = sum_sort_top_1(fer_1_tgt_src_map[tgt_idx], 1)
    
    align_str = subword_to_word_merge_fwd_rev(fer_1_src_tgt_map, fer_1_tgt_src_map, sub2word_map_src, sub2word_map_tgt)
        
    return align_str


def ot_mbr_get_maps(src_emb_, tgt_emb_):
    P_dict = {}
    # config_sink = {'method': 'sink', 'epsilon': 0.1}
    # P_sink = get_ot_map(src_emb_, tgt_emb_, config_sink)
    # P_dict['f2f'] = P_sink
    
    # # NOTE: p_sink is for soft partial-to-partial mapping
    # config_p_sink = {'method': 'p_sink', 'epsilon': 0.1, 'relax_ratio': 2.0}
    # P_p_sink = get_ot_map(src_emb_, tgt_emb_, config_p_sink)
    # P_dict['p2p'] = P_p_sink
    
    # # NOTE: p_sink_f2p is for full-to-partial mapping
    # config_p_sink_f2p = {'method': 'p_sink_f2p', 'epsilon': 0.1, 'relax_ratio': 2.0}
    # P_p_sink_f2p = get_ot_map(src_emb_, tgt_emb_, config_p_sink_f2p)
    # P_dict['f2p'] = P_p_sink_f2p
    
    # # NOTE: p_sink_p2f is for soft partial-to-full mapping
    # config_p_sink_p2f = {'method': 'p_sink_p2f', 'epsilon': 0.1, 'relax_ratio': 2.0}
    # P_p_sink_p2f = get_ot_map(src_emb_, tgt_emb_, config_p_sink_p2f)
    # P_dict['p2f'] = P_p_sink_p2f
    
    # NOTE: p_sink_f2f_fwd is for foward align with full-to-full sinkhorn
    config_p_sink_f2f_fwd = {'method': 'p_sink_f2p_fwd', 'epsilon': 0.05}
    P_p_sink_f2f_fwd = get_ot_map(src_emb_, tgt_emb_, config_p_sink_f2f_fwd)
    P_dict['fwd'] = P_p_sink_f2f_fwd
    
    # NOTE: p_sink_f2f_rev is for reverse align with full-to-full sinkhorn
    config_p_sink_f2f_rev = {'method': 'p_sink_p2f_rev', 'epsilon': 0.05}     
    P_p_sink_f2f_rev = get_ot_map(src_emb_, tgt_emb_, config_p_sink_f2f_rev)
    P_dict['rev'] = P_p_sink_f2f_rev
    
    return P_dict

def pmi_align_token(out_src, out_tgt, sub2word_map_src, sub2word_map_tgt, use_sim_align=False):

    def pmi_matrix(out_src, out_tgt):

        sim = torch.matmul(out_src, out_tgt.transpose(-1, -2))
        if use_sim_align:
            return sim
        sim = torch.softmax(sim.view(-1), dim=0).view(sim.size())

        probs_src = torch.sum(sim, dim=1)
        probs_tgt = torch.sum(sim, dim=0)

        repeat_probs_src = probs_src.unsqueeze(1).expand(-1, sim.size(-1))
        repeat_probs_tgt = probs_tgt.repeat(sim.size(0), 1)
        scores = torch.log(sim) - torch.log(repeat_probs_tgt) - torch.log(repeat_probs_src)

        scores = (scores - scores.min()) / (scores.max() - scores.min())

        return scores

    out_src.div_(torch.norm(out_src, dim=-1).unsqueeze(-1))
    out_tgt.div_(torch.norm(out_tgt, dim=-1).unsqueeze(-1))

    dot_prod = pmi_matrix(out_src, out_tgt)
    
    # dot_prod = torch.matmul(out_src, out_tgt.transpose(-1, -2))
    
    # dot_prod = 1 - dot_prod
    # # dot_prod, _ = _sinkhorn_iter(dot_prod, 30)
    # dot_prod = ot.sinkhorn(ot.unif(dot_prod.size(0), type_as=dot_prod), ot.unif(dot_prod.size(1), type_as=dot_prod), dot_prod, reg=0.1)
    # print(p.shape)
    
    # A = iter_max(dot_prod.numpy())
    # align_words = np.nonzero(A)
    # align_words2 = set()
    # for i in range(len(align_words[0])):
    #   align_words2.add((sub2word_map_src[align_words[0][i]], sub2word_map_tgt[align_words[1][i]]))

    argmax_srctgt = torch.argmax(dot_prod, dim=-1)
    argmax_tgtsrc = torch.argmax(dot_prod, dim=-2)

    align_words_srctgt = set()
    align_words_tgtsrc = set()
    for i, j in enumerate(argmax_srctgt):
        align_words_srctgt.add(((sub2word_map_src[i], sub2word_map_tgt[j])))

    for i, j in enumerate(argmax_tgtsrc):
        align_words_tgtsrc.add((sub2word_map_src[j], sub2word_map_tgt[i]))

    align_words = align_words_srctgt.intersection(align_words_tgtsrc)
    # align_words = align_words_tgtsrc
    # indices = sinkhorn(dot_prod)
    # align_words = set()
    # for p in indices:
    #   align_words.add( (sub2word_map_src[p[0]], sub2word_map_tgt[p[1]]) )

    alignStr = ""
    for p in align_words:
        alignStr += str(p[0]) + "-" + str(p[1]) + " "

    return alignStr




def pmi_align_word(src_rep, mt_rep, shift_one, use_sim_align=False):
    src_rep = torch.stack(src_rep, dim=0).cuda()
    mt_rep = torch.stack(mt_rep, dim=0).cuda()

    def pmi_matrix(out_src, out_tgt):
        out_src.div_(torch.norm(out_src, dim=-1).unsqueeze(-1))
        out_tgt.div_(torch.norm(out_tgt, dim=-1).unsqueeze(-1))

        sim = torch.matmul(out_src, out_tgt.transpose(-1, -2))
        if use_sim_align:
            return sim
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

    dot_prod = pmi_matrix(out_src, out_tgt)
    
    # A = iter_max(dot_prod.numpy())
    # align_words = np.nonzero(A)
    # align_words2 = set()
    # for i in range(len(align_words[0])):
    #   align_words2.add((sub2word_map_src[align_words[0][i]], sub2word_map_tgt[align_words[1][i]]))

    argmax_srctgt = torch.argmax(dot_prod, dim=-1)
    argmax_tgtsrc = torch.argmax(dot_prod, dim=-2)

    align_words_srctgt = set()
    align_words_tgtsrc = set()
    for i, j in enumerate(argmax_srctgt):
        i = int(i) + shift_one
        j = int(j) + shift_one
        align_words_srctgt.add((str(i), str(j)))

    for i, j in enumerate(argmax_tgtsrc):
        i = int(i) + shift_one
        j = int(j) + shift_one
        align_words_tgtsrc.add((str(j), str(i)))

    align_words = align_words_srctgt.intersection(align_words_tgtsrc)

    # indices = sinkhorn(dot_prod)
    # align_words = set()
    # for p in indices:
    #   align_words.add( (sub2word_map_src[p[0]], sub2word_map_tgt[p[1]]) )

    alignStr = ""
    for p in align_words:
        alignStr += str(p[0]) + "-" + str(p[1]) + " "

    return alignStr


def iter_max(sim_matrix: np.ndarray, max_count: int = 2) -> np.ndarray:
    alpha_ratio = 0.9
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


def extract_wa_from_pi_xi(pi, xi):
    m, n = pi.size()
    forward = torch.eye(n)[pi.argmax(dim=1)]
    backward = torch.eye(m)[xi.argmax(dim=0)]
    inter = forward * backward.transpose(0, 1)
    ret = []
    for i in range(m):
        for j in range(n):
            if inter[i, j].item() > 0:
                ret.append((i, j))
    return ret


def sinkhorn(sim, num_iter=2):
    pred_wa = []
    pi, xi = _sinkhorn_iter(sim, num_iter)
    pred_wa_i_wo_offset = extract_wa_from_pi_xi(pi, xi)
    for src_idx, trg_idx in pred_wa_i_wo_offset:
        pred_wa.append((src_idx, trg_idx))
    return pred_wa


def _sinkhorn_iter(S, num_iter=2):
    if num_iter <= 0:
        return S, S
    # assert num_iter >= 1
    assert S.dim() == 2
    # S[S <= 0] = 1e-6
    S[S <= 0].fill_(1e-6)
    # pi = torch.exp(S*10.0)
    pi = S
    xi = pi
    for i in range(num_iter):
        pi_sum_over_i = pi.sum(dim=0, keepdim=True)
        xi = pi / pi_sum_over_i
        xi_sum_over_j = xi.sum(dim=1, keepdim=True)
        pi = xi / xi_sum_over_j
    return pi, xi