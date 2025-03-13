import sys
[sys.path.append(i) for i in ['.', '..']]

import os, pickle, argparse, copy, random, json, re, sys
import logging, zipfile, subprocess, multiprocessing, shutil, math, codecs, glob
from _collections import defaultdict
from tqdm import tqdm
import pandas as pd
from wordfreq import word_frequency

import numpy as np
logger = logging.getLogger(__name__)

#########################
#### load Edinburgh ######
#########################

def load_annotation(annotation, sure_and_possible, sent1, sent2):
    s1_words = sent1.split()
    s2_words = sent2.split()

    alignments = []
    for align in annotation['S']:
        sid = align[0]
        assert sid >= 0 and sid < len(s1_words)

        for tid in align[1]:
            assert tid >= 0 and tid < len(s2_words)
            alignments.append('{0}-{1}'.format(sid, tid))

    if sure_and_possible:
        for align in annotation['P']:
            sid = align[0]
            assert sid >= 0 and sid < len(s1_words)

            for tid in align[1]:
                assert tid >= 0 and tid < len(s2_words)
                alignments.append('{0}-{1}'.format(sid, tid))

    return alignments


def split_Edinburgh_corpus(portion, sent1_list, sent2_list, alignment_list):
    random.seed(42)
    idx_list = list(range(len(sent1_list)))
    random.shuffle(idx_list)
    ids = idx_list[:514] if portion == "train" else idx_list[514:]
    return zip(*[(sent1_list[i], sent2_list[i], alignment_list[i]) for i in ids])

def load_Edinburgh_corpus(file_path, portion="test"):
    sent1_list = []
    sent2_list = []
    alignment_list = []
    sure_and_possible = False
    with open(file_path) as f:
        df = json.load(f)

    for item in df['paraphrases']:
        sent1 = item['S']['string']
        sent2 = item['T']['string']
        align_A, align_C = [], []
        if 'A' in item['annotations']:
            align_A = load_annotation(item['annotations']['A'], sure_and_possible, sent1, sent2)
            align_A = set(align_A)
            alignment = align_A
        if 'C' in item['annotations']:
            align_C = load_annotation(item['annotations']['C'], sure_and_possible, sent1, sent2)
            align_C = set(align_C)
            alignment = align_C
        if len(align_A) > 0 and len(align_C) > 0:
            alignment = align_A & align_C

        sent1_list.append(sent1)
        sent2_list.append(sent2)
        alignment_list.append(alignment)

    # unified format
    alignment_list = [" ".join(item) for item in alignment_list]
    if portion != "test":
        return split_Edinburgh_corpus(portion, sent1_list, sent2_list, alignment_list)

    return sent1_list, sent2_list, alignment_list

####################
#### load RTE ######
####################

def split_MSR_RTE_corpus(portion, sent1_list, sent2_list, alignment_list):
    random.seed(42)
    idx_list = list(range(len(sent1_list)))
    random.shuffle(idx_list)
    ids = idx_list[:600] if portion == "train" else idx_list[600:]
    return zip(*[(sent1_list[i], sent2_list[i], alignment_list[i]) for i in ids])

def load_MSR_RTE_corpus(corpus_dir, portion="test"):

    all_annotators = []

    for file_path in glob.glob(corpus_dir + '*.align.txt'):
        examples = load_MSR_RTE_file(file_path)
        all_annotators.append(examples)

    # Consolidate annotations
    sents1, sents2, alignments = [], [], []
    for id in range(len(all_annotators[0][0])):
        sents1.append(all_annotators[0][0][id])
        sents2.append(all_annotators[0][1][id])
        # Take annotations for which at least 2 annotators agreed
        alignment_ab = all_annotators[0][2][id] & all_annotators[1][2][id]
        alignment_ac = all_annotators[0][2][id] & all_annotators[2][2][id]
        alignment_bc = all_annotators[1][2][id] & all_annotators[2][2][id]
        alignment = set(alignment_ab | alignment_bc | alignment_ac)
        alignments.append(alignment)


    # unified format
    alignments = [" ".join(item) for item in alignments]
    if portion != "test":
        return split_MSR_RTE_corpus(portion, sents1, sents2, alignments)

    return sents1, sents2, alignments

def load_MSR_RTE_file(file_path, portion="test"):
    sure_and_possible = False
    sent1_list = []
    sent2_list = []
    alignment_list = []
    p = re.compile(r'\({([\sp\d]+?)/ / }\)?')

    with codecs.open(file_path, encoding='utf-8') as f:
        lines = [l.strip().replace('\ufeff', '') for l in f.readlines()] # Remove BOF char that could not removed by nkf
        for lid in range(len(lines)):
            if lid % 3 == 0:
                sent1 = lines[lid + 1]
                s1_words = sent1.split()

                # convert alignment
                sent2_alignments = lines[lid + 2]
                a_lists = p.findall(sent2_alignments)
                s2_words = p.sub(r'', sent2_alignments).split()
                assert len(a_lists) == len(s2_words)
                alignments = []
                for tid, alist in enumerate(a_lists[1:]):  # 0 is NULL
                    for align in alist.strip().split():
                        if align[0] == 'p':
                            sid = int(align[1:]) - 1  # 1-base indexing -> 0-base
                            assert sid < len(s1_words) and sid >= 0
                            if sure_and_possible:
                                alignments.append('{0}-{1}'.format(sid, tid))
                            else:
                                pass
                        else:
                            sid = int(align) - 1  # 1-base indexing -> 0-base
                            assert sid < len(s1_words) and sid >= 0
                            alignments.append('{0}-{1}'.format(sid, tid))

                sent1_list.append(sent1)
                sent2_list.append(' '.join(s2_words[1:]))  # 0 is NULL
                alignment_list.append(set(alignments))



    return sent1_list, sent2_list, alignment_list

def read_Word_Alignment_Dataset(filename, sure_and_possible=False, transpose=False):
    data = []
    for line in open(filename):
        ID, sent1, _, sent2, _, _, _, sure_align, poss_align, *useless = line.strip('\n').split('\t')
        my_dict = {}
        my_dict['source'] = sent1
        my_dict['target'] = sent2
        my_dict['sureAlign'] = sure_align
        my_dict['possibleAlign'] = poss_align
        data.append(my_dict)
    sent1_list = []
    sent2_list = []
    alignment_list = []
    for i in range(len(data)):
        if transpose:
            source = data[i]['target']
            target = data[i]['source']
        else:
            source = data[i]['source']
            target = data[i]['target']
        alignment = data[i]['sureAlign']
        if sure_and_possible:
            alignment += ' ' + data[i]['possibleAlign']
        my_label = []
        for item in alignment.split():  # reverse the alignment
            i, j = item.split('-')
            if transpose:
                my_label.append(str(j) + '-' + str(i))
            else:
                my_label.append(str(i) + '-' + str(j))
        alignment = ' '.join(my_label)
        sent1_list.append(source.lower().split())
        sent2_list.append(target.lower().split())
        alignment_list.append(alignment)

    # unified format
    sent1_list, sent2_list = [" ".join(l) for l in sent1_list], [" ".join(l) for l in sent2_list]

    return sent1_list, sent2_list, alignment_list

def iter_load_MWA_corpus(corpus_dir, only_ds_name=False):

    for filename in glob.glob(corpus_dir + '/*/*.tsv'):
        _, name = os.path.split(filename)
        ds_name, portion = name.split(".")[0].split("-")
        ds_name = f"mwa_{ds_name}_{portion}"

        if only_ds_name:
            yield ds_name
        else:
            sent1_list, sent2_list, alignment_list = read_Word_Alignment_Dataset(filename)

            yield ds_name, sent1_list, sent2_list, alignment_list

########################
#### OTAlign Eval ######
########################
def compute_score(pred, gold):
    gold = set(gold.split())
    if len(pred) > 0:
        precision = len(gold & pred) / len(pred)
    else:
        if len(gold) == 0:
            precision = 1
        else:
            precision = 0
    if len(gold) > 0:
        recall = len(gold & pred) / len(gold)
    else:
        if len(pred) == 0:
            recall = 1
        else:
            recall = 0
    if len(pred & gold) == len(gold) and len(pred & gold) == len(pred):
        acc = 1
    else:
        acc = 0

    return precision, recall, acc

def get_aligned_indices(alignments):
    aligned_s1_indices, aligned_s2_indices = set(), set()
    for a in alignments:
        i_j = re.split('[\-p]', a)
        aligned_s1_indices.add(int(i_j[0]))
        aligned_s2_indices.add(int(i_j[1]))
    return aligned_s1_indices, aligned_s2_indices

def compute_null_score(pred, gold, sent1, sent2):
    gold = set(gold.split())
    sent1 = sent1.split()
    sent2 = sent2.split()
    gold_s1_indices, gold_s2_indices = get_aligned_indices(gold)
    gold_null_s1_indices = set(range(len(sent1))) - gold_s1_indices
    gold_null_s2_indices = set(range(len(sent2))) - gold_s2_indices

    aligned_s1_indices, aligned_s2_indices = get_aligned_indices(pred)
    pred_null_s1_indices = set(range(len(sent1))) - aligned_s1_indices
    pred_null_s2_indices = set(range(len(sent2))) - aligned_s2_indices

    if len(pred_null_s1_indices) + len(pred_null_s2_indices) > 0:
        precision = (len(gold_null_s1_indices & pred_null_s1_indices) + len(
            gold_null_s2_indices & pred_null_s2_indices)) / (len(pred_null_s1_indices) + len(pred_null_s2_indices))
    else:
        if len(gold_null_s1_indices) + len(gold_null_s2_indices) == 0:
            precision = 1
        else:
            precision = 0
    if len(gold_null_s1_indices) + len(gold_null_s2_indices) > 0:
        recall = (len(gold_null_s1_indices & pred_null_s1_indices) + len(
            gold_null_s2_indices & pred_null_s2_indices)) / (len(gold_null_s1_indices) + len(gold_null_s2_indices))
    else:
        if len(pred_null_s1_indices) + len(pred_null_s2_indices) == 0:
            recall = 1
        else:
            recall = 0

    if len(gold_null_s1_indices & pred_null_s1_indices) == len(gold_null_s1_indices) and len(
            gold_null_s1_indices & pred_null_s1_indices) == len(pred_null_s1_indices) and len(
        gold_null_s2_indices & pred_null_s2_indices) == len(gold_null_s2_indices) and len(
        gold_null_s2_indices & pred_null_s2_indices) == len(pred_null_s2_indices):
        acc = 1.0
    else:
        acc = 0.0

    return precision, recall, acc

def compute_total_score(pred, gold, sent1, sent2):
    gold = set(gold.split())
    sent1 = sent1.split()
    sent2 = sent2.split()
    gold_s1_indices, gold_s2_indices = get_aligned_indices(gold)
    gold_null_s1_indices = set(range(len(sent1))) - gold_s1_indices
    gold_null_s2_indices = set(range(len(sent2))) - gold_s2_indices

    aligned_s1_indices, aligned_s2_indices = get_aligned_indices(pred)
    pred_null_s1_indices = set(range(len(sent1))) - aligned_s1_indices
    pred_null_s2_indices = set(range(len(sent2))) - aligned_s2_indices

    precision = (len(gold & pred) + len(gold_null_s1_indices & pred_null_s1_indices) + len(
        gold_null_s2_indices & pred_null_s2_indices)) / (
                        len(pred) + len(pred_null_s1_indices) + len(pred_null_s2_indices))
    recall = (len(gold & pred) + len(gold_null_s1_indices & pred_null_s1_indices) + len(
        gold_null_s2_indices & pred_null_s2_indices)) / (
                     len(gold) + len(gold_null_s1_indices) + len(gold_null_s2_indices))

    if len(pred & gold) == len(gold) and len(pred & gold) == len(pred):
        acc = 1
    else:
        acc = 0

    return precision, recall, acc

def evaluate_alignments(outputs):

    precision_all = []
    recall_all = []
    accuracy_all = []
    null_precision_all = []
    null_recall_all = []
    null_accuracy_all = []
    total_precision_all = []
    total_recall_all = []
    total_accuracy_all = []
    # align_thresh = self.determine_align_thresh(outputs)
    golds = outputs['gold']
    sents1 = outputs['s1_sents']
    sents2 = outputs['s2_sents']
    preds = outputs['pred']

    for bidx in range(len(golds)):
        p, r, acc = compute_score(preds[bidx], golds[bidx])
        null_p, null_r, null_acc = compute_null_score(preds[bidx], golds[bidx], sents1[bidx], sents2[bidx])
        total_p, total_r, total_acc = compute_total_score(preds[bidx], golds[bidx], sents1[bidx], sents2[bidx])

        precision_all.append(p)
        recall_all.append(r)
        accuracy_all.append(acc)
        null_precision_all.append(null_p)
        null_recall_all.append(null_r)
        null_accuracy_all.append(null_acc)
        total_precision_all.append(total_p)
        total_recall_all.append(total_r)
        total_accuracy_all.append(total_acc)

    precision = sum(precision_all) / len(precision_all)
    recall = sum(recall_all) / len(recall_all)
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    accuracy = sum(accuracy_all) / len(accuracy_all)

    null_precision = sum(null_precision_all) / len(null_precision_all)
    null_recall = sum(null_recall_all) / len(null_recall_all)
    if null_precision + null_recall > 0:
        null_f1 = 2 * null_precision * null_recall / (null_precision + null_recall)
    else:
        null_f1 = 0.0
    null_accuracy = sum(null_accuracy_all) / len(null_accuracy_all)

    total_precision = sum(total_precision_all) / len(total_precision_all)
    total_recall = sum(total_recall_all) / len(total_recall_all)
    if total_precision + total_recall > 0:
        total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall)
    else:
        total_f1 = 0.0
    total_accuracy = sum(total_accuracy_all) / len(total_accuracy_all)

    logs = {"precision": precision, "recall": recall, "f1": f1, "exact_match": accuracy,
            "null_precision": null_precision, "null_recall": null_recall, "null_f1": null_f1,
            "null_exact_match": null_accuracy, "total_precision": total_precision, "total_recall": total_recall,
            "total_f1": total_f1, "total_exact_match": total_accuracy}

    logs = {k: 100* v for k, v in logs.items()}
    return logs

#############################
#### one-to-many align ######
#############################
def _eval_one2many(sent_lst, gold_tags_lst, pred_tags_lst):
    rint = random.randint(1, 100000)
    output_file, score_file = f"/tmp/output.{rint}.txt", f"/tmp/score.{rint}.txt"
    fout = open(output_file, 'w')

    for words, gold_lst, pred_lst in zip(sent_lst, gold_tags_lst, pred_tags_lst):
        s = "\n".join([f"{w} {g} {p}" for w, g, p in zip(words, gold_lst, pred_lst)])
        fout.write(f"{s}\n\n")
    fout.close()

    os.system("perl %s -r < %s > %s" % ("utils/conlleval", output_file, score_file))
    eval_lines = [l.rstrip() for l in open(score_file)]
    # print(open(output_file).read())
    # print(open(score_file).read())
    f1_score = float(eval_lines[1].strip().split()[-1])

    return f1_score

def _get_one2many_tag_lst(words, align_str, n, is_bwd=False):
    tag_lst = ["O"] * n
    group_dict = defaultdict(list)

    for item in align_str.split():
        pair = re.split("[p\-]", item)
        if is_bwd: pair = pair[::-1]
        group_dict[int(pair[0])].append(int(pair[1]))

    lst = [(k, sorted(lst)) for k, lst in group_dict.items() if len(lst) > 2]

    for i1, i2_lst in lst:
        for idx in range(len(i2_lst)):
            is_b = not idx or (idx and i2_lst[idx] != i2_lst[idx-1]+1)
            tag_lst[i2_lst[idx]-1] = f"B-{words[i1-1]}" if is_b else f"I-{words[i1-1]}"
            # tag_lst[i2_lst[idx] - 1] = f"B-MULTI" if is_b else f"I-MULTI"

    return tag_lst

def evaluate_one2many_alignment(src_text_lst, tgt_text_lst, gold_lst, pred_lst):
    gold_fwd_tags_lst, pred_fwd_tags_lst = [], []
    gold_bwd_tags_lst, pred_bwd_tags_lst = [], []
    sent_fwd_lst, sent_bwd_lst = [], []

    for src_text, tgt_text, gold, pred in zip(src_text_lst, tgt_text_lst, gold_lst, pred_lst):
        sent_fwd_lst.append(src_text.split())
        sent_bwd_lst.append(tgt_text.split())
        n_src, n_tgt = len(sent_fwd_lst[-1]), len(sent_bwd_lst[-1])

        tag_fwd_gold_lst = _get_one2many_tag_lst(sent_fwd_lst[-1], gold, n_tgt, is_bwd=False)
        tag_bwd_gold_lst = _get_one2many_tag_lst(sent_bwd_lst[-1], gold, n_src, is_bwd=True)

        tag_fwd_pred_lst = _get_one2many_tag_lst(sent_fwd_lst[-1], pred, n_tgt, is_bwd=False)
        tag_bwd_pred_lst = _get_one2many_tag_lst(sent_bwd_lst[-1], pred, n_src, is_bwd=True)

        gold_fwd_tags_lst.append(tag_fwd_gold_lst)
        pred_fwd_tags_lst.append(tag_fwd_pred_lst)
        gold_bwd_tags_lst.append(tag_bwd_gold_lst)
        pred_bwd_tags_lst.append(tag_bwd_pred_lst)

    f1_fwd_score = _eval_one2many(sent_fwd_lst, gold_fwd_tags_lst, pred_fwd_tags_lst)
    f1_bwd_score = _eval_one2many(sent_bwd_lst, gold_bwd_tags_lst, pred_bwd_tags_lst)

    f1_score = (f1_fwd_score + f1_bwd_score) / 2
    scores = {"one2many_src_f1": f1_fwd_score, "one2many_tgt_f1": f1_bwd_score, "one2many_f1": f1_score}
    return scores


#########################
#### hallucination ######
#########################
LANG_MAP = {"eng": "en", "arb": "ar", "rus": "ru","zho": "zh", "spa": "es", "deu": "de",
            "mni": "mni",  "kas": "ks", "yor": "yo"}
ar_char = "\u0621-\u064a\ufb50-\ufdff\ufe70-\ufefc"

class HallExample():

    def __init__(self):
        self.ds_name = None

        self.data_source = None
        self.src_lang = ""
        self.tgt_lang = ""

        self.src_text = ""
        self.tgt_text = ""

        self.omit_span_idx_lst = []
        self.hall_span_idx_lst = []

        self.lbl_hall, self.lbl_omit = None, None

        self.src_word_lst, self.src_pos_lst, self.src_word_offset_mapping = [], [], []
        self.src_ner_lst, self.src_ner_mention_dict = [], []
        self.src_wf_lst, self.src_sw_lst = [], []

        self.tgt_word_lst, self.tgt_pos_lst, self.tgt_word_offset_mapping, = [], [], []
        self.tgt_ner_lst, self.tgt_ner_mention_dict = [], []
        self.tgt_wf_lst, self.tgt_sw_lst = [], []

        self.util_data_dict = {}

def _word_break_tokenizer(text):
    from camel_tools.tokenizers.word import _TOKENIZE_RE

    word_lst, word_offset_mapping, char2word_mapping = [], [], {}

    for word_idx, m in enumerate(_TOKENIZE_RE.finditer(text)):
        word_lst.append(m.group())
        word_offset_mapping.append((m.start(), m.end()))
        for i in range(m.start(), m.end()):
            char2word_mapping[i] = word_idx

    return word_lst, word_offset_mapping, char2word_mapping



def _get_selected_span_index(text, text_with_span):
    prefix, suffix = "<<<", ">>>"
    text_with_span = text_with_span.replace("<<<'", "<<<")
    # text_with_span = text_with_span.replace("Хм! Хм>>>", "Хм! xм>>>")
    skip_word = ["Whitehall", "AAAAAAAAAAAAAAAASS", " Puerto ",
                 "2001:B011:300D:2DC7:B4EE:1774:7DD9:5EA1", "€237", "─┤", "◄"]
    # if text.count("!") > 20 or any(w in text for w in skip_word): return [], True

    prefix_idx_lst = [m.start() for m in re.finditer(prefix, text_with_span)]
    suffix_idx_lst = [m.start() for m in re.finditer(suffix, text_with_span)]

    index_lst = []
    rm = 0
    is_data_issue = text_with_span.replace(prefix, "").replace(suffix, "") != text

    for i, (begin, end) in enumerate(zip(prefix_idx_lst, suffix_idx_lst)):
        index_lst.append((begin-rm, end-len(prefix)-rm))
        rm += len(prefix) + len(suffix)
        s1 = text_with_span[begin+len(prefix):end]
        s2 = text[index_lst[-1][0]:index_lst[-1][1]]
        if not is_data_issue:
            assert s1 == s2


    return index_lst, is_data_issue

def _load_spacy_models(models_dir):
    import spacy
    spacy_dict = {}

    for lang in LANG_MAP.values():
        spacy_path = os.path.join(models_dir, f"spacy_models/{lang}_core_web_sm-3.5.0")
        if not os.path.exists(spacy_path):
            spacy_path = os.path.join(models_dir, f"spacy_models/{lang}_core_news_sm-3.5.0")
        if not os.path.exists(spacy_path): continue

        # spacy_path = os.path.split(spacy_path)[1]
        spacy_dict[lang] = spacy.load(spacy_path)
        skip_lst = ["cannot", "dont", "gotta", "gonna", "im"]
        if lang == 'en':
            spacy_dict[lang].tokenizer.rules = \
                {key: value for key, value in spacy_dict[lang].tokenizer.rules.items()
                 if all(k not in key for k in skip_lst) }

    spacy_dict["ar"] = _load_camel_models()

    return spacy_dict


def _load_stop_word_dict():
    stop_word_dict = {}

    from spacy.lang.en import stop_words
    stop_word_dict["en"] = stop_words.STOP_WORDS

    from spacy.lang.de import stop_words
    stop_word_dict["de"] = stop_words.STOP_WORDS

    from spacy.lang.ru import stop_words
    stop_word_dict["ru"] = stop_words.STOP_WORDS

    from spacy.lang.es import stop_words
    stop_word_dict["es"] = stop_words.STOP_WORDS

    from spacy.lang.zh import stop_words
    stop_word_dict["zh"] = stop_words.STOP_WORDS

    from spacy.lang.ar import stop_words
    stop_word_dict["ar"] = stop_words.STOP_WORDS

    return stop_word_dict

def _load_camel_models():
    from camel_tools.disambig.mle import MLEDisambiguator
    from camel_tools.tagger.default import DefaultTagger
    from camel_tools.ner import NERecognizer

    ner = NERecognizer.pretrained()
    mle = MLEDisambiguator.pretrained()
    tagger = DefaultTagger(mle, 'pos')

    return {"camel_pos": tagger, "camel_ner": ner}

def _annotate_sentence(spacy_dict, stop_word_dict, lang, sentence):
    if lang == "ar":
        return _annotate_sentence_with_camel(spacy_dict["ar"], stop_word_dict[lang], sentence)
    elif lang in spacy_dict:
        return _annotate_sentence_with_spacy(spacy_dict, stop_word_dict, lang, sentence)
    else:
        return  _annotate_sentence_low_resource(sentence)



def _annotate_sentence_with_spacy(spacy_dict, stop_word_dict, lang, sent):
    doc = spacy_dict[lang](sent)

    word_lst, pos_lst = [w.text for w in doc], [w.pos_ for w in doc]
    word_offset_mapping = [(w.idx, w.idx + len(w.text)) for w in doc]

    char2word_mapping = {}
    for word_idx, (begin, end) in enumerate(word_offset_mapping):
        for i in range(begin, end):
            char2word_mapping[i] = word_idx

    ner_lst = ["O"] * len(doc)
    ner_mention_dict = {}
    for ent in doc.ents:

        for i in range(ent.start, ent.end):
            prefix = "B-" if i == ent.start else "I-"
            ner_lst[i] = f"{prefix}{ent.label_}"

        ner_mention_dict[(ent.start, ent.end)] = ent.label_

    sw_lst = [w.lower() in stop_word_dict for w in word_lst]
    wf_lst = [word_frequency(w.lower(), lang) for w  in word_lst]

    assert len(ner_lst) == len(word_lst)
    return {"word_lst": word_lst, "sw_lst": sw_lst, "wf_lst": wf_lst,
            "word_offset_mapping": word_offset_mapping, "char2word_mapping": char2word_mapping,
            "pos_lst": pos_lst, "ner_lst": ner_lst, "ner_mention_dict": ner_mention_dict}

def _annotate_sentence_with_camel(camel_models, stop_word_dict, sentence):
    word_lst, word_offset_mapping, char2word_mapping = _word_break_tokenizer(sentence)
    pos_lst = camel_models["camel_pos"].tag(word_lst)
    ner_lst = camel_models["camel_ner"].predict_sentence(word_lst)
    ner_mention_dict = {}
    label, begin, end = "", None, None

    for i, tag in enumerate(ner_lst):
        if tag.startswith('B-'):
            if label:
                ner_mention_dict[(begin, end)] = label
            begin = i
            end = i + 1
            label = tag[2:]
        elif tag.startswith('I-') and label:
            end += 1

        else:
            if label:
                ner_mention_dict[(begin, end)] = label
                label, begin, end = "", None, None

    if label:
        ner_mention_dict[(begin, end)] = label

    sw_lst = [w in stop_word_dict for w in word_lst]
    wf_lst = [word_frequency(w, "ar") for w in word_lst]

    return {"word_lst": word_lst,  "sw_lst": sw_lst, "wf_lst": wf_lst,
            "char2word_mapping": char2word_mapping, "word_offset_mapping": word_offset_mapping,
            "pos_lst": pos_lst, "ner_lst": ner_lst, "ner_mention_dict": ner_mention_dict}

def _annotate_sentence_low_resource(sentence):
    word_lst, word_offset_mapping, char2word_mapping = _word_break_tokenizer(sentence)

    tmp_lst = ["O"] * len(word_lst)
    pos_lst, ner_lst = copy.deepcopy(tmp_lst), copy.deepcopy(tmp_lst)
    ner_mention_dict = {}
    sw_lst, wf_lst = [False] * len(word_lst), [0.0] * len(word_lst)

    return {"word_lst": word_lst,  "sw_lst": sw_lst, "wf_lst": wf_lst,
            "char2word_mapping": char2word_mapping, "word_offset_mapping": word_offset_mapping,
            "pos_lst": pos_lst, "ner_lst": ner_lst, "ner_mention_dict": ner_mention_dict}

def get_word_lvl_lbl(span_idx_lst, word_offset_mapping):
    tags = [0] * len(word_offset_mapping)
    if not span_idx_lst:
         return tags

    char_dict = {}
    for word_idx, (begin, end) in enumerate(word_offset_mapping):
        for i in range(begin, end):
            char_dict[i] = word_idx

    word_id_set = set()
    for begin, end in span_idx_lst:
        word_id_set.update([char_dict.get(i, -1) for i in range(begin, end)])

    for word_idx in word_id_set:
        if word_idx == -1: continue
        tags[word_idx] = 1

    return tags

def load_raw_hall(data_dir, only_ds_name=False):
    # ds_dict = {"hall_deen": _load_hall_deen(data_dir)}

    ds_dict = _load_halomi(data_dir)

    if only_ds_name:
        return list(ds_dict.keys())

    #ds_dict = {k: v for k, v in ds_dict.items() if k == "halomi_en:de"}
    # ds_dict = {"hall_deen": ds_dict["hall_deen"]}

    print("Annotate Hall data")
    spacy_dict = _load_spacy_models(os.path.join(data_dir, "..", "models"))
    stop_word_dict = _load_stop_word_dict()

    for ds_name, example_lst in ds_dict.items():

        for i, example in enumerate(tqdm(example_lst)):

            dico = _annotate_sentence(spacy_dict, stop_word_dict, example.src_lang, example.src_text)
            for k, v in dico.items():
                example.util_data_dict[f"src_{k}"] = v

            dico = _annotate_sentence(spacy_dict, stop_word_dict, example.tgt_lang, example.tgt_text)
            for k, v in dico.items():
                example.util_data_dict[f"tgt_{k}"] = v

            hall_span_idx_lst = example.util_data_dict.get("hall_span_idx_lst", [])
            example.util_data_dict["tgt_wl_lbl"] = get_word_lvl_lbl(hall_span_idx_lst, example.util_data_dict["tgt_word_offset_mapping"])

            omit_span_idx_lst = example.util_data_dict.get("omit_span_idx_lst", [])
            example.util_data_dict["src_wl_lbl"] = get_word_lvl_lbl(omit_span_idx_lst, example.util_data_dict["src_word_offset_mapping"])

            ds_dict[ds_name][i] = example



    if "halomi_en:ru" in ds_dict:ds_dict["halomi_en:ru"].pop(44)
    if "halomi_en:es" in ds_dict:ds_dict["halomi_en:es"].pop(27)

    return ds_dict



def _load_hall_deen(data_dir):


    fn = open(os.path.join(data_dir, "hallucinations_deen_w_stats_and_scores.pkl"), 'rb')
    dataset_stats = pickle.load(fn)
    annotated = pd.read_csv( os.path.join(data_dir, "annotated_corpus.csv"))

    df = pd.merge(
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
    example_lst = []
    skip_lst = ["€237", "90/1998", "€100,000", "€2.45"]
    for idx, row in enumerate(df.to_dict('records')):
        example = HallExample()
        example.data_source = "hall_deen"

        example.src_lang, example.tgt_lang = "de", "en"
        example.src_text = re.sub('\s+', ' ', row["src"]).strip()
        example.tgt_text = re.sub('\s+', ' ', row["mt"]).strip()

        # if any (w in example.src_text or w in example.tgt_text for w in skip_lst):
        #     continue
        for k, v in row.items():
            example.util_data_dict[k] = v

        if idx not in [1051]:
            example_lst.append(example)


    return example_lst

def _load_halomi(data_dir):
    df = pd.read_csv(
        os.path.join("halomi_release_v2/data", "halomi_core.tsv"),
        sep="\t",
        keep_default_na=False,
    )
    ds_dict = defaultdict(list)
    data_issue_counter = 0
    skip_lang_set = {"kas_Deva", "mni_Beng", "yor_Latn"}

    for row in df.to_dict('records'):
        # if row["src_lang"] in skip_lang_set or row["tgt_lang"] in skip_lang_set: continue

        example = HallExample()
        example.ds_name = row["data_source"]

        example.src_lang = LANG_MAP[row["src_lang"].split("_")[0]]
        example.tgt_lang = LANG_MAP[row["tgt_lang"].split("_")[0]]
        example.src_text = re.sub('\s+', ' ', row["src_text"]).strip()
        example.tgt_text = re.sub('\s+', ' ', row["mt_text"]).strip()

        example.util_data_dict["omit_span_idx_lst"], is_omit_data_issue = _get_selected_span_index(example.src_text, row["omit_spans"].strip())
        example.util_data_dict["hall_span_idx_lst"], is_hall_data_issue = _get_selected_span_index(example.tgt_text, row["hall_spans"].strip())
        example.util_data_dict["lbl_hall"], example.util_data_dict["lbl_omit"] = \
            int(row["class_hall"].split("_")[0]), int(row["class_omit"].split("_")[0])

        if is_omit_data_issue or is_hall_data_issue:
            data_issue_counter += 1
        else:
            ds_name = f"halomi_{example.src_lang}:{example.tgt_lang}"
            ds_dict[ds_name].append(example)

    return ds_dict



##################
#### others ######
###################
def untokenize(text):
    """
    Untokenizing a text undoes the tokenizing operation, restoring
    punctuation and spaces to the places that people expect them to be.
    Ideally, `untokenize(tokenize(text))` should be identical to `text`,
    except for line breaks.
    """

    step1 = text.replace("`` ", '"').replace(" ''", '"').replace('. . .',  '...')
    step2 = step1.replace(" ( ", " (").replace(" ) ", ") ")
    step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)
    step4 = re.sub(r' ([.,:;?!%]+)$', r"\1", step3)
    step5 = step4.replace(" '", "'").replace(" n't", "n't").replace(
         "can not", "cannot")
    step6 = step5.replace(" ` ", " '")
    return step6.strip()

def _encode_ds(dico, features, d_dir):
    encoded_dataset = arrow_dataset.Dataset.from_dict(dico, features=features)
    encoded_dataset.save_to_disk(d_dir)


def iterative_dataset_save(tmp_dir, tmp_dataset_dir, data_dict, features, batch_size=1000):

    tmp_tmp_dir = os.path.join(tmp_dir, "tmp")
    if os.path.exists(tmp_tmp_dir):
        shutil.rmtree(tmp_tmp_dir)
    os.mkdir(tmp_tmp_dir)

    key_lst = list(data_dict.keys())
    sample_num = len(data_dict[key_lst[0]])

    os.mkdir(tmp_dataset_dir)
    # write dataset individually
    dir_lst = []
    attrs_lst = []
    for i in tqdm(range(0, sample_num, batch_size)):
        dico = {key: data_dict[key][i:i+batch_size] for key in key_lst}
        d_dir = os.path.join(tmp_tmp_dir, str(i // batch_size))
        attrs_lst.append((dico, features, d_dir))
        dir_lst.append(d_dir)

    max_process = len(attrs_lst)
    p = multiprocessing.Pool(max_process)
    p.starmap(_encode_ds, attrs_lst)
    p.close()
    # combine datasets
    _data_files = []
    counter = 0
    for i, d_dir in enumerate(dir_lst):

        file_lst = [filename for filename in os.listdir(d_dir) if filename.endswith(".arrow")]
        file_lst.sort(key=lambda x:int(x.split("-")[1]))
        for filename in file_lst:
            src = os.path.join(d_dir, filename)
            dst = os.path.join(tmp_dataset_dir, "dataset_%s.arrow" % counter)
            os.rename(src, dst)
            _data_files.append({"filename": "dataset_%s.arrow" % counter})
            counter += 1

        # load state.json and update it
        state = json.load(open(os.path.join(d_dir, "state.json")))
        state["_data_files"] = _data_files

        src = os.path.join(d_dir, "dataset_info.json")
        shutil.copyfile(src, os.path.join(tmp_dataset_dir, "dataset_info.json"))

        fout = open(os.path.join(tmp_dataset_dir, "state.json"), 'w')
        json.dump(state, fout, ensure_ascii=False, indent=4)
        fout.close()


