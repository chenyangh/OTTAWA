import torch
import numpy as np

def pmi_align(src_rep, mt_rep):
    src_rep = torch.stack(src_rep, dim=0).cuda()
    mt_rep = torch.stack(mt_rep, dim=0).cuda()
    
    def pmi_matrix(out_src, out_tgt):
        out_src.div_(torch.norm(out_src, dim=-1).unsqueeze(-1))
        out_tgt.div_(torch.norm(out_tgt, dim=-1).unsqueeze(-1))
            
        sim = torch.matmul(out_src, out_tgt.transpose(-1, -2))
        sim = torch.softmax(sim.view(-1), dim=0).view(sim.size())
        
        probs_src = torch.sum(sim, dim = 1)
        probs_tgt = torch.sum(sim, dim = 0)
        
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
        i = int(i) + 1
        j = int(j) + 1
        align_words_srctgt.add( (str(i), str(j))) 
    
    for i, j in enumerate(argmax_tgtsrc):
        i = int(i) + 1
        j = int(j) + 1
        align_words_tgtsrc.add( (str(j), str(i)) )
    
    align_words = align_words_srctgt.intersection(align_words_tgtsrc)
    
    # indices = sinkhorn(dot_prod)
    # align_words = set()
    # for p in indices:
    #   align_words.add( (sub2word_map_src[p[0]], sub2word_map_tgt[p[1]]) )

    alignStr = ""
    for p in align_words:
        alignStr += str(p[0]) + "-" + str(p[1]) + " "
    
    return alignStr        


def iter_max(sim_matrix: np.ndarray, max_count: int=2) -> np.ndarray:
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
  S[S<=0].fill_(1e-6)
  # pi = torch.exp(S*10.0)
  pi = S
  xi = pi
  for i in range(num_iter):
    pi_sum_over_i = pi.sum(dim=0, keepdim=True)
    xi = pi / pi_sum_over_i
    xi_sum_over_j = xi.sum(dim=1, keepdim=True)
    pi = xi / xi_sum_over_j
  return pi, xi