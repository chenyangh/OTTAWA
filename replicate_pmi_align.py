import numpy as np
from tqdm import tqdm
from my_ot_algorithm import *
from other_ot_algorithms import *
from align_utils import *
from transformers import BertTokenizer, BertModel


tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained("bert-base-multilingual-cased")



src = 'en'
tgt = 'fr'
num_samples = len(open(f'PMI-Align/data/{src}-{tgt}/text.{src}', 'r').readlines())

results = []


gold_aligen_list = []

with tqdm(total=num_samples, desc='Processing') as pbar:
    for test_line_id in range(num_samples):
        
        
        gold_alignment = open(f'PMI-Align/data/{src}-{tgt}/gold.{src}-{tgt}.aligned', 'r').readlines()[test_line_id]
        src_text = open(f'PMI-Align/data/{src}-{tgt}/text.{src}', 'r').readlines()[test_line_id].strip()
        tgt_text = open(f'PMI-Align/data/{src}-{tgt}/text.{tgt}', 'r',).readlines()[test_line_id].strip()
        
        
        if len(src_text.split()) == 0:
            print('Finished @', test_line_id)
            print('Processed', len(results), 'samples')
            break
        
        try:
            src_emb_ = get_word_embeddings(src_text, model, tokenizer)
            tgt_emb_ = get_word_embeddings(tgt_text, model, tokenizer)
        except Exception as e:
            print('error emb @', test_line_id)
            continue
        
        pmi_align_str = pmi_align(src_emb_, tgt_emb_)

        results.append(pmi_align_str)
        
        # For progress bar
        eval_result = evaluate_corpus_level(pmi_align_str, gold_aligen_list)
        f1 = eval_result['F1 Score']
        aer = eval_result['AER']
        precision = eval_result['Precision']
        recall = eval_result['Recall']
        avg_f1_str = f'{f1:.4f}'
        avg_aer_str = f'{aer:.4f}'
        avg_pre_str = f'{precision:.4f}'
        avg_rec_str = f'{recall:.4f}'
        pbar.set_postfix({'F1': avg_f1_str, 'AER': avg_aer_str, 'Precision': avg_pre_str, 'Recall': avg_rec_str})
        pbar.update(1)

    
    result = evaluate_corpus_level(results, gold_aligen_list)
    print("Average F1 Score:", result['F1 Score'])
    print("Average AER:", result['AER'])
    print("Average Precision:", result['Precision'])
    print("Average Recall:", result['Recall'])

    