import numpy as np
import torch
import torch.nn.functional as F
from OTAlign.src.util import min_max_scaling
from ot.backend import get_backend

def subword_to_word_merge_fwd_rev(srctgt_map, tgtsrc_map, sub2word_map_src, sub2word_map_tgt):
    align_words_srctgt = set()
    align_words_tgtsrc = set()
    for i, j in srctgt_map.items():
        if j == -1: 
            continue
        align_words_srctgt.add(((sub2word_map_src[i], sub2word_map_tgt[j])))

    for i, j in tgtsrc_map.items():
        if j == -1: 
            continue
        align_words_tgtsrc.add((sub2word_map_src[j], sub2word_map_tgt[i]))

    align_words = align_words_srctgt.intersection(align_words_tgtsrc)

    alignStr = ' '.join([str(p[0]) + "-" + str(p[1]) for p in align_words])
    return alignStr

def update_top_choices(top_choices, new_choice):
    # top_choices and new_choice are dictionaries
    for choice in new_choice:
        if choice not in top_choices:
            top_choices[choice] = [new_choice[choice]]
        else:
            top_choices[choice].append(new_choice[choice])
    return top_choices
        
def sum_sort_top_1(idx_val_lst_dict, k=1):
    sorted_list = list(sorted(idx_val_lst_dict.items(), key=lambda x: sum(x[1]), reverse=True)[:k])
    return sorted_list[0][0] if len(sorted_list) > 0 else -1
    

def get_top_p_indices(tensor, p=0.1):
    # Sort the tensor in descending order
    sorted_tensor, sorted_indices = torch.sort(tensor, descending=True)
    
    # Compute the cumulative sum
    cumulative_sum = torch.cumsum(sorted_tensor, dim=0)
    
    # Find the index where the cumulative sum exceeds p
    cutoff_index = torch.sum(cumulative_sum <= p).item()
    
    # Return the top p indices
    return sorted_indices[:cutoff_index]

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
    s1_word_embeddigs = F.normalize(s1_word_embeddigs)
    s2_word_embeddigs = F.normalize(s2_word_embeddigs)
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


def map_original_to_tokenized(x1, x2):
    mapping = {}
    tokenized_index = 0
    original_index = 0

    while original_index < len(x2) and tokenized_index < len(x1):
        original_word = x2[original_index]
        subword_sequence = ''

        indices = []
        while tokenized_index < len(x1) and (subword_sequence != original_word):
            subword = x1[tokenized_index].lstrip('##')
            if subword_sequence + subword == original_word[:len(subword_sequence + subword)]:
                subword_sequence += subword
                indices.append(tokenized_index)
                tokenized_index += 1
            else:
                break
        if not indices:
            tokenized_index += 1
        else:
            mapping[original_index] = indices
        original_index += 1
    return mapping


def map_original_to_sentencepiece(x1, x2):
    mapping = {}
    tokenized_index = 0
    original_index = 0

    # Replace with the actual underscore character used in your SentencePiece tokenization
    sentence_piece_underscore = '▁'

    while original_index < len(x2) and tokenized_index < len(x1):
        original_word = x2[original_index]
        subword_sequence = ''

        indices = []
        while tokenized_index < len(x1) and (subword_sequence != original_word):
            subword = x1[tokenized_index].lstrip(sentence_piece_underscore)
            if subword and (subword_sequence + subword == original_word[:len(subword_sequence) + len(subword)]):
                subword_sequence += subword
                indices.append(tokenized_index)
            tokenized_index += 1

        # Special handling for tokens that are just the SentencePiece underscore or punctuation
        if not indices and x1[tokenized_index - 1].strip(sentence_piece_underscore) == '':
            indices.append(tokenized_index - 1)

        if indices:
            mapping[original_index] = indices
        original_index += 1

    return mapping


def get_word_embeddings(sentence, model, tokenizer):
    # Tokenize the sentence and get corresponding IDs
    inputs = tokenizer(sentence, return_tensors="pt")
    # tokens = inputs.tokens()[1:-1]
    tokens = tokenizer.tokenize(sentence)
    mapping = map_original_to_tokenized(tokens, sentence.split())

    # Get BERT embeddings for each token
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    embeddings = outputs['hidden_states'][8].squeeze(0)[1:-1]

    assert len(embeddings) == len(tokens)
    
    word_embeddings = []
    for original_idx in range(len(sentence.split())):
        subword_indices = mapping[original_idx]
        subword_embeddings = [embeddings[idx] for idx in subword_indices]
        averaged_embedding = torch.mean(torch.stack(subword_embeddings), dim=0)
        word_embeddings.append(averaged_embedding)

    assert len(word_embeddings) == len(sentence.split(" "))
    return word_embeddings


def get_predicted_alignment(P, method='l2r', thres=None, is_fwd=False):
    # Apply argmax along each row
    null_idx = P.shape[1] - 1
    if method == 'l2r':
        alignment_indices = np.argmax(P, axis=1)
        
        # Generate alignment string
        if is_fwd:
            alignment_string = ' '.join(f'{i+1}-{j+1}' for i, j in enumerate(alignment_indices) if j != null_idx and sum(P[i]) > 0)
        else:
            alignment_string = ' '.join(f'{j+1}-{i+1}' for i, j in enumerate(alignment_indices) if j != null_idx and sum(P[i]) > 0)
        
    elif method == 'r2f':
        alignment_indices = np.argmax(P, axis=0)
        # Generate alignment string
        if is_fwd:
            alignment_string = ' '.join(f'{j+1}-{i+1}' for i, j in enumerate(alignment_indices) if j != null_idx and sum(P[:, i]) > 0)
        else:
            alignment_string = ' '.join(f'{i+1}-{j+1}' for i, j in enumerate(alignment_indices) if j != null_idx and sum(P[:, i]) > 0)
        
    elif method == 'thres':
        assert thres is not None
        P_thres = P.copy()
        thres = P_thres[:, :-1].sum(axis=1)[:, np.newaxis] * thres  # can be designed
        bin_P = (P_thres >= thres).astype(int)
        alignments = []
        # Iterate over the matrix
        for i in range(bin_P.shape[0]):  # For each source (row)
            if P_thres[i].argmax(-1) == null_idx:
                continue
            for j in range(bin_P.shape[1]):  # For each target (column)
                if j == null_idx:
                    continue
                if bin_P[i, j] == 1:
                    # Record the alignment, adjusting index to be one-based
                    if is_fwd:
                        alignments.append(f"{i+1}-{j+1}")
                    else:
                        alignments.append(f"{j+1}-{i+1}")
        # Join all alignments into a string
        alignment_string = ' '.join(alignments)
    
    return alignment_string

def rank_align_pairs(align):
    align_pairs = align.split()
    align_pairs = [pair.split('-') for pair in align_pairs if 'p' not in pair]
    align_pairs = [(int(pair[0]), int(pair[1])) for pair in align_pairs]
    align_pairs = sorted(align_pairs, key=lambda x: x[0] * 1000 + x[1])
    return align_pairs


def get_joint_alignments(fwd_align, rev_align, method='union'):
    fwd_align_pairs = rank_align_pairs(fwd_align)
    rev_align_pairs = rank_align_pairs(rev_align)
    
    if method == 'union':
        align_pairs = list(set(fwd_align_pairs).union(set(rev_align_pairs)))
    elif method == 'intersection':
        align_pairs = list(set(fwd_align_pairs).intersection(set(rev_align_pairs)))
    else:
        raise NotImplementedError
    
    align_pairs = sorted(align_pairs, key=lambda x: x[0])
    align_string = ' '.join([f'{pair[0]}-{pair[1]}' for pair in align_pairs])
    return align_string


def evaluate_corpus_level(pred_list, align_list):
    # # Example usage
    # alignment_string = "5-6 3-4 25p22 5-7 2p3 25p21 4-5 11-14 12-13 13-15 27p20 9-11 25p23 8-8 22-25 20-27 17-18 1-1 28-28 26p20 18-19 6p9 1-2 7-10 10-12 21-24 15-17 14-16 23-26"
    # predicted_string = "5-6 3-4 25-22 ..."  # Replace with your model's predicted alignments

    def parse_alignments(alignment_string):
        sure_alignments = set()
        possible_alignments = set()

        # Split the string into individual alignments
        alignments = alignment_string.split()

        for alignment in alignments:
            if 'p' in alignment:
                # Possible alignment
                aligned_words = tuple(map(int, alignment.split('p')))
                possible_alignments.add(aligned_words)
            else:
                # Sure alignment
                aligned_words = tuple(map(int, alignment.split('-')))
                sure_alignments.add(aligned_words)

        return sure_alignments, possible_alignments

    def compare(predicted_alignments, sure_alignments, possible_alignments):
        a_and_s = len(predicted_alignments.intersection(sure_alignments))
        a_and_p = len(predicted_alignments.intersection(possible_alignments))
        
        return a_and_p, a_and_s

    sum_a_and_s = 0
    sum_a_and_p = 0
    sum_len_a = 0
    sum_len_s = 0
    
    for pred, align in zip(pred_list, align_list):
        # Parse alignments
        sure_alignments, possible_alignments = parse_alignments(align)
        predicted_alignments = parse_alignments(pred)[0]
        possible_alignments = possible_alignments.union(sure_alignments)
        
        a_and_p, a_and_s = compare(predicted_alignments, sure_alignments, possible_alignments)
        sum_a_and_s += a_and_s
        sum_a_and_p += a_and_p
        sum_len_a += len(predicted_alignments)
        sum_len_s += len(sure_alignments)
    
    precision = sum_a_and_p / sum_len_a
    recall = sum_a_and_s / sum_len_s
    aer = 1 - (sum_a_and_s + sum_a_and_p) / (sum_len_a + sum_len_s)
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    return {"F1 Score": f1_score, "AER": aer, "Precision": precision, "Recall": recall}


def format_alignments(alignments):
        """
        Formats the set of alignment tuples into a string where each alignment is represented
        by 'source-target' and source/target indices are one-based.
        """
        formatted = ' '.join(f'{src+1}-{tgt+1}' for src, tgt in sorted(alignments))
        return formatted
    
    
def evaluate_sent_level(predicted_string, alignment_string):
    # # Example usage
    # alignment_string = "5-6 3-4 25p22 5-7 2p3 25p21 4-5 11-14 12-13 13-15 27p20 9-11 25p23 8-8 22-25 20-27 17-18 1-1 28-28 26p20 18-19 6p9 1-2 7-10 10-12 21-24 15-17 14-16 23-26"
    # predicted_string = "5-6 3-4 25-22 ..."  # Replace with your model's predicted alignments
    def parse_alignments(alignment_string):
        sure_alignments = set()
        possible_alignments = set()

        # Split the string into individual alignments
        alignments = alignment_string.split()

        for alignment in alignments:
            if 'p' in alignment:
                # Possible alignment
                aligned_words = tuple(map(int, alignment.split('p')))
                possible_alignments.add(aligned_words)
            else:
                # Sure alignment
                aligned_words = tuple(map(int, alignment.split('-')))
                sure_alignments.add(aligned_words)

        return sure_alignments, possible_alignments

    def calculate_f1(predicted_alignments, sure_alignments, possible_alignments):
        a_and_s = len(predicted_alignments.intersection(sure_alignments))
        a_and_p = len(predicted_alignments.intersection(possible_alignments))
        prec = a_and_p / len(predicted_alignments) if len(predicted_alignments) > 0 else 0
        rec = a_and_s / len(sure_alignments) if len(sure_alignments) > 0 else 0

        if prec + rec == 0:
            return 0
        return 2 * (prec * rec) / (prec + rec), prec, rec

    def calculate_aer(predicted_alignments, sure_alignments, possible_alignments):
        a_and_s = len(predicted_alignments.intersection(sure_alignments))
        a_and_p = len(predicted_alignments.intersection(possible_alignments))
        return 1 - (a_and_s + a_and_p) / (len(predicted_alignments) + len(sure_alignments))

    
    # Parse alignments
    sure_alignments, possible_alignments = parse_alignments(alignment_string)
    predicted_alignments = parse_alignments(predicted_string)[0]  # Assuming model predicts only sure alignments

    # Calculate metrics
    f1_score, prec, rec = calculate_f1(predicted_alignments, sure_alignments, possible_alignments.union(sure_alignments))
    aer = calculate_aer(predicted_alignments, sure_alignments, possible_alignments.union(sure_alignments))

    # print("F1 Score:", f1_score)
    # print("AER:", aer)
    return {"F1 Score": f1_score, "AER": aer, "Precision": prec, "Recall": rec}


