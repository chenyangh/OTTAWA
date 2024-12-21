import sys, itertools
[sys.path.append(i) for i in ['.', '..']]

from collections import Counter

import torch
from utils.utils import *
from utils.align_methods import *
from utils.hall_methods import *
from align_utils import *
from models.ot_hall_utils import *

from transformers import (
    AutoTokenizer,
    AutoModel
)
import h5py


MODEL_NAME_LST = ["bert-base-multilingual-cased", "xlm-roberta-base",
                  "xlm-align-base", "xlm-roberta-large", "xlm-roberta-xl", "xlm-mlm-100-1280", "bert-base-uncased",
                  "AWESOME", "LaBSE"]
SHIFT_ONE_DS = {"de-en",  "en-cs", "en-fr", "en-hi", "ro-en"}

LAYER_IDX_LST = [5, 7, 8, 9, 10, 11]
# LAYER_IDX_LST = [16, 17, 18, 19, 20, 21, 22, 23]  # xlm-roberta-large
# LAYER_IDX_LST = [16, 18, 28, 30, 32, 34] # xlm-roberta-xl
# LAYER_IDX_LST = [11, 12, 13, 14, 15] # xlm-roberta-xl

LABSE_LAYER_IDX = 10

######################################################

#import pickle as pkl 
avg_emb_list = None #  pkl.load(open("labase_avg_emb_list.pkl", "rb"))

# all_results = []

class Aligner():

    def __init__(self, args, ds_name, src_lang, tgt_lang):
        self.args = args
        self.ds_name = ds_name
        self.tokenize_style = args.tokenize_style
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.model_name = args.model_name
        self.model_path = os.path.join(args.model_dir, self.model_name)
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=True)

        self.util_data_dict = {}
        self.ds_src, self.ds_tgt = None, None
        self.hall_eval_data = defaultdict(list)


    def dump_emb(self, attrs_dict):

        for key in ["ds_name", "src_lang", "tgt_lang", "util_data_dict"]:
            setattr(self, key, attrs_dict[key])

        self.ds_src, self.ds_tgt = AlignDataset(), AlignDataset()
        key_lst = ["word_lst", "pos_lst", "ner_lst", "sw_lst", "wf_lst",
                   "mention_dict", "word_offset_mapping", "char2word_mapping", "wl_lbl"]

        for k, v in attrs_dict["util_data_dict"].items():
            if not k.startswith("src_") and not k.startswith("tgt_"):
                self.hall_eval_data[k].append(v)

        if "util_data_dict" in attrs_dict:
            for d in ["src", "tgt"]:
                for key in key_lst:
                    att = f"{d}_{key}"

                    val = attrs_dict["util_data_dict"].get(att, None)
                    if val and d == "src" and hasattr(self.ds_src, key):
                        setattr(self.ds_src, key, val)

                    if val and d == "tgt" and hasattr(self.ds_tgt, key):
                        setattr(self.ds_tgt, key, val)

        self._dump_emb(self.ds_src, attrs_dict["src_text_lst"], self.src_lang)
        self._dump_emb(self.ds_tgt, attrs_dict["tgt_text_lst"], self.tgt_lang)

    def _dump_emb(self, align_ds, text_lst, lang):
        assert not (self.args.tokenize_style == "standard" and self.args.model_name in ["xlm-roberta-base"])

        print("Batch Tokenization")
        align_ds.ds_name = self.ds_name
        align_ds.lang = lang
        align_ds.text_lst = text_lst
        align_ds, input_dict = _tokenize(self.args, self.tokenizer, align_ds, text_lst)

        print("Predict Embedding")
        new_text_lst = [" ".join(words) for words in align_ds.word_lst]
        text_lst = align_ds.text_lst if self.args.tokenize_style == "standard" else  new_text_lst
        _dump_embedding(self.args, self.ds_name, lang, self.model_path, input_dict, text_lst)

        align_ds.set_cache_path(self.args.tmp_dir, self.model_name, self.tokenize_style)

        return align_ds
    
    def run_hall(self, args, ds_name):
        if not self.ds_src.h5_fin_dict:
            self.ds_src.open_cache_dict()
            self.ds_tgt.open_cache_dict()
        print(args.align_method)
        n = len(self.ds_src.text_lst)
        gold_lst, pred_lst = [], []
        shift_one = int(self.ds_name in SHIFT_ONE_DS)


        error_type = args.error_type # 'hall' | 'omi' | 'entity'
        for sent_idx in tqdm(range(n)):
            emb_src = self.ds_src.get_word_emb(sent_idx, args.layer_idx, merge_method=args.merge_method)
            emb_tgt = self.ds_tgt.get_word_emb(sent_idx, args.layer_idx, merge_method=args.merge_method)
            if args.ds_type == "hall":
                src_word_feature_dict, src_token_feature_dict = self.ds_src.get_feature_dict(sent_idx)
                tgt_word_feature_dict, tgt_token_feature_dict = self.ds_tgt.get_feature_dict(sent_idx)

            if args.align_method == "pmi_align_word":
                # assert args.merge_method != "none"
                if error_type == 'hall':
                    target_category = 'lbl_hall'
                elif error_type == 'omi':
                    target_category = 'lbl_omit'
                    exclude_hall = True
                    if exclude_hall:
                        not_hall = self.hall_eval_data["lbl_hall"][0][sent_idx] == 1
                        if not not_hall:
                            continue
                else:
                    raise ValueError(f"error_type {error_type} is not implemented!!!")

                pos_cond_4 = self.hall_eval_data[target_category][0][sent_idx] == 4
                pos_cond_3 = self.hall_eval_data[target_category][0][sent_idx] == 3
                pos_cond_2 = self.hall_eval_data[target_category][0][sent_idx] == 2
                pos_cond =  pos_cond_4 or pos_cond_3 or pos_cond_2
                # tgt_label = int(pos_cond)
                tgt_label = self.hall_eval_data[target_category][0][sent_idx]
                
                avg_emb = None                    
                    
                kwargs = {"src_rep": [torch.tensor(l) for l in emb_src],
                          "mt_rep": [torch.tensor(l) for l in emb_tgt],
                          "shift_one": shift_one,
                          "use_sim_align": args.use_sim_align,
                          "src_feature_dict": src_word_feature_dict,
                          "tgt_feature_dict": tgt_word_feature_dict,
                          "src_token_feature_dict": src_token_feature_dict,
                          "tgt_token_feature_dict": tgt_token_feature_dict,  
                          "avg_emb": avg_emb,
                          "error_type": error_type,
                          # "align_method": "itermax" # pmi, ot, argmax, itermax, pot
                          }

                
                ot_score = ot_align_fwd_rev(**kwargs)
                
                
                gold_lst.append(tgt_label)
                pred_lst.append(ot_score)
                
                src_text = self.ds_src.text_lst[sent_idx]
                tgt_text = self.ds_tgt.text_lst[sent_idx]                
                           
            else:
                raise ValueError(f"align_method {args.align_method} is not implemented!!!")
            
        # NOTE: add the hall evaluation here
       
        # result = compute_auc(pred_lst, gold_lst)
        result = percent_correct_pairs(gold_lst, pred_lst)
        return result
    

class AlignDataset():
    def __init__(self):
        self.ds_name = None
        self.lang = None
        self.text_lst = []
        self.word_lst = []

        self.input_ids_lst = None
        self.token2word_mapping = []
        self.word2token_mapping = []

        self.word_offset_mapping = []
        self.char2word_mapping = []

        self.pos_lst, self.ner_lst, self.sw_lst, self.wf_lst = [], [], [], []
        self.word_offset_mapping, mention_dict = [], []

        self.cache_path_dict, self.h5_fin_dict = {}, {}

        self.wl_lbl = []

    def rmv_example_by_index(self, rmv_set):
        for key in dir(self):
            if isinstance(getattr(self, key), list):
                att_lst = getattr(self, key)
                lst = [att_lst[i] for i in range(len(att_lst)) if i not in rmv_set]
                setattr(self, key, lst)

    def set_cache_path(self, tmp_dir, model_name, tokenize_style):
        # init the cache dir names
        for layer_idx in LAYER_IDX_LST:
            key = f"emb_{self.ds_name}_{self.lang}_{model_name}_{tokenize_style}_{layer_idx}.h5py"
            self.cache_path_dict[layer_idx] = os.path.join(tmp_dir, key)

    def open_cache_dict(self):
        for key, cache_path in self.cache_path_dict.items():
            if not os.path.exists(cache_path): continue
            self.h5_fin_dict[key] = h5py.File(cache_path, 'r')

    def get_word_emb(self, sent_idx, layer_idx, merge_method="mean"):
        vec = self.h5_fin_dict[layer_idx][str(sent_idx)]
        word_embeddings = []

        if merge_method == "mean":
            for token_idx_lst in self.word2token_mapping[sent_idx]:
                arr = np.mean(np.asarray([vec[i] for i in token_idx_lst]), axis=0)
                word_embeddings.append(arr)
        elif merge_method == "max":
            for token_idx_lst in self.word2token_mapping[sent_idx]:
                arr = np.max(np.asarray([vec[i] for i in token_idx_lst]), axis=0)
                word_embeddings.append(arr)
        elif merge_method == "first":
            word_embeddings = [vec[token_idx_lst[0]] for token_idx_lst in self.word2token_mapping[sent_idx]]
        elif merge_method == "none":
            token_idx_lst = sorted(list(self.token2word_mapping[sent_idx].keys()))
            word_embeddings = [vec[token_idx] for token_idx in token_idx_lst]
        else:
            raise ValueError(f"{merge_method} token merging method is not implemented!!!")

        return word_embeddings

    def get_sub2word_map(self, sent_idx):
        lst = sorted(self.token2word_mapping[sent_idx].items(), key= lambda x:x[0])
        _, sub2word_map = zip(*lst)
        if self.ds_name in SHIFT_ONE_DS:
            sub2word_map = [i+1 for i in sub2word_map]
        return sub2word_map

    def get_feature_dict(self, sent_idx):
        freq_ths = 1e-03 # TODO Chenyang

        word_feature_dict = {"is_ner": [tag != "O" for tag in self.ner_lst[sent_idx]],
                            "is_low_freq": [wf < freq_ths for wf in self.wf_lst[sent_idx]],
                            "is_sw_lst": self.sw_lst[sent_idx],
                            "is_punct": [tag.lower().startswith("punc") for tag in self.pos_lst[sent_idx]]
                            }

        _, idx_lst = zip(*sorted(self.token2word_mapping[sent_idx].items(), key= lambda x:x[0]))

        token_feature_dict = {k: [feat_lst[idx] for idx in idx_lst] for k, feat_lst in word_feature_dict.items()}

        return word_feature_dict, token_feature_dict


def _get_raw2word_mapping(text):
    begin = 0
    words, word_offset_mapping = [], []
    for m in re.finditer(' ', text):
        words.append(text[begin:m.start()])
        word_offset_mapping.append((begin, m.start()))
        begin = m.start()+1
    if begin != len(text):
        words.append(text[begin:len(text)])
        word_offset_mapping.append((begin, len(text)))

    char2word_mapping = {}
    for idx, (i, j) in enumerate(word_offset_mapping):
        for k in range(i, j):
            char2word_mapping[k] = idx

    return words, word_offset_mapping, char2word_mapping

def _get_token2word_mapping(char2word_mapping, token_offset_mapping):
    token2word_counter = defaultdict(list)

    for token_idx, (i, j) in enumerate(token_offset_mapping):
        for k in range(i, j):
            token2word_counter[token_idx].append(char2word_mapping.get(k, None))

    token2word_mapping = {}
    word2token_mapping = defaultdict(list)
    for token_idx, lst in token2word_counter.items():
        token2word_mapping[token_idx] = sorted(Counter(lst).items(), key= lambda x:x[1])[-1][0]
        word2token_mapping[token2word_mapping[token_idx]].append(token_idx)

    for word_idx, lst in word2token_mapping.items():
        word2token_mapping[word_idx] = sorted(lst)

    word2token_mapping = sorted(word2token_mapping.items(), key= lambda x:x[0])
    word2token_mapping = [lst for _, lst in word2token_mapping]

    word_num = len(set(char2word_mapping.values()))
    # print(char2word_mapping)
    # print(token_offset_mapping)
    # print(word2token_mapping)
    # print(len(word2token_mapping), word_num)
    assert len(word2token_mapping) == word_num

    return token2word_mapping, word2token_mapping


def _tokenize(args, tokenizer, align_ds, text_lst):
    if args.tokenize_style == "standard":
        align_ds, input_dict =  _tokenize_standard(tokenizer, align_ds, text_lst)
    elif args.tokenize_style == "pmi":
        if args.ds_type == "hall":
            new_text_lst = [" ".join(words) for words in align_ds.word_lst]
            align_ds, input_dict =  _tokenize_pmi(args, tokenizer, align_ds, new_text_lst, False)
        else:
            align_ds, input_dict = _tokenize_pmi(args, tokenizer, align_ds, text_lst)
    else:
        raise ValueError(f"Not implemented tokenization Style {args.tokenize_style}!!!!")\

    # filter long sequence
    rmv_set = set()
    for i, input_ids in enumerate(input_dict["input_ids"]):
        if len(input_ids) > 256:
            print(len(input_ids), i)
            rmv_set.add(i)
    #
    # input_dict = {k: [v[j] for j in range(len(v)) if j not in rmv_set] for k, v in input_dict.items()}
    # align_ds.rmv_example_by_index(rmv_set)

    return align_ds, input_dict

def _tokenize_standard(tokenizer, align_ds, text_lst):

    dico = tokenizer.batch_encode_plus(text_lst,
                                       add_special_tokens=True,
                                       return_offsets_mapping=True)

    align_ds.input_ids_lst = dico["input_ids"]
    is_preproc = bool(len(align_ds.word_lst))

    print("Set Text Attribute")
    for k, (token_offset_mapping, text) in enumerate(zip(dico["offset_mapping"], text_lst)):
        if not is_preproc:
            word_lst, word_offset_mapping, char2word_mapping = _get_raw2word_mapping(text)
            align_ds.word_lst.append(word_lst)
            align_ds.word_offset_mapping.append(word_offset_mapping)
            align_ds.char2word_mapping.append(char2word_mapping)

        # print(text)
        # print(align_ds.word_lst[k])
        token2word_mapping, word2token_mapping = \
            _get_token2word_mapping(align_ds.char2word_mapping[k], token_offset_mapping)
        align_ds.token2word_mapping.append(token2word_mapping)
        align_ds.word2token_mapping.append(word2token_mapping)

    input_dict = copy.deepcopy(dico)
    del input_dict["offset_mapping"]

    return align_ds, input_dict


def _tokenize_pmi(args, tokenizer, align_ds, text_lst, use_raw=False):
    shift_one = int(args.model_name in ["bert-base-multilingual-cased", "bert-base-uncased", "LaBSE"])
    is_preproc = bool(len(align_ds.word_lst))
    if is_preproc and use_raw:
        idx_lst = [list(range(len(words))) for words in align_ds.word_lst]
    else:
        if not align_ds.word_lst:
            align_ds.word_lst = [sentence.split() for sentence in text_lst]

        lst = [[(idx, word) for word in words] for idx, words in enumerate(align_ds.word_lst)]
        idx_lst, word_lst = zip(*[item for items in lst for item in items])
        assert all(len(l1) == len(l2) for l1, l2 in zip(align_ds.word_lst, align_ds.ner_lst))

    group_dict = defaultdict(list)
    for j, idx in enumerate(idx_lst):
        group_dict[idx].append(j)

    input_ids_lst = tokenizer.batch_encode_plus(word_lst,
                                                add_special_tokens=False,
                                                return_offsets_mapping=False)["input_ids"]

    align_ds.input_ids_lst = []
    input_dict = defaultdict(list)

    for _, lst in sorted(group_dict.items(), key=lambda x:x[0]):
        w2t_map, t2w_map = [], {}
        input_ids = []

        for word_idx, j in enumerate(lst):
            w2t_map.append([k + len(input_ids) + shift_one for k in range(len(input_ids_lst[j]))])
            t2w_map.update({j: word_idx for j in w2t_map[word_idx]})

            input_ids += input_ids_lst[j]
        # print(input_ids)
        align_ds.input_ids_lst.append(input_ids)

        align_ds.word2token_mapping.append(w2t_map)
        align_ds.token2word_mapping.append(t2w_map)

        dico = tokenizer.prepare_for_model(input_ids,
                                           model_max_length=tokenizer.model_max_length,
                                           truncation=True)
        for k, v in dico.items():
            input_dict[k].append(v)

    return align_ds, input_dict

def _batch_gen(input_dict, batch_size=64, device="cuda:0"):
    n = len(input_dict["input_ids"])
    key_lst = input_dict.keys()
    for i in range(0, n, batch_size):
        max_len = max(len(l) for l in input_dict["input_ids"][i:i+batch_size])
        output_dict = {}

        for key in key_lst:
            output_dict[key] = [l + [0] * (max_len-len(l)) for l in input_dict[key][i:i+batch_size]]
            output_dict[key] = torch.as_tensor(output_dict[key], dtype=torch.long).to(device)

        yield output_dict

def _dump_embedding(args, ds_name, lang, model_path, input_dict, text_lst):
    if args.model_name in ["LaBSE"]:
        return _dump_embedding_labse(args, ds_name, lang, model_path, text_lst)
    else:
        return _dump_embedding_nolabase(args, ds_name, lang, model_path, input_dict)

def _dump_embedding_labse(args, ds_name, lang, model_path, text_lst):
    from sentence_transformers import SentenceTransformer
    try:
        model = SentenceTransformer(model_path)
    except:
        model_path = "sentence-transformers/LaBSE"
        model = SentenceTransformer(model_path)
        
    model.eval()

    key = f"emb_{ds_name}_{lang}_{args.model_name}_{args.tokenize_style}_%s.h5py"
    cache_path = os.path.join(args.tmp_dir, key)

    h5_fout_dict = {LABSE_LAYER_IDX: h5py.File(cache_path % LABSE_LAYER_IDX, 'w')}

    embeddings = model.encode(text_lst, output_value='token_embeddings')
    for counter, emb in enumerate(embeddings):
        h5_fout_dict[LABSE_LAYER_IDX][str(counter)] = emb.detach().cpu().numpy()

    for h5_fout in h5_fout_dict.values():
        h5_fout.close()

    del model


def _dump_embedding_nolabase(args, ds_name, lang, model_path, input_dict):
    model = AutoModel.from_pretrained(model_path).to(args.device)
    model.eval()

    h5_fout_dict = {}
    key = f"emb_{ds_name}_{lang}_{args.model_name}_{args.tokenize_style}_%s.h5py"
    cache_path = os.path.join(args.tmp_dir, key)
    for layer_idx in LAYER_IDX_LST:
        h5_fout_dict[layer_idx] = h5py.File(cache_path % layer_idx, 'w')

    gen = _batch_gen(input_dict, device=args.device)
    counter = defaultdict(int)
    for inputs in tqdm(gen):
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        for layer_idx in LAYER_IDX_LST:
            embeddings = outputs['hidden_states'][layer_idx].detach().cpu().numpy()
            n = embeddings.shape[0]
            for i in range(n):
                seq_len = sum(input_dict["attention_mask"][counter[layer_idx]])
                h5_fout_dict[layer_idx][str(counter[layer_idx])] = embeddings[i, :seq_len]
                counter[layer_idx] += 1

    for h5_fout in h5_fout_dict.values():
        h5_fout.close()

    del model

def _iter_load_raw_ds(args, only_ds_name=False):
    if args.ds_type == "PMI-Align":
        data_dir = os.path.join(args.data_dir, "PMI-Align")
        for ds_name in os.listdir(data_dir):
            # if ds_name != "en-fr": continue
            if ds_name.count("-") != 1 or not os.path.isdir(os.path.join(data_dir, ds_name)): continue
            if only_ds_name:
                yield ds_name
            else:
                src_lang, tgt_lang = ds_name.split("-")
                d_dir = os.path.join(args.data_dir, "PMI-Align", ds_name)
                

                # read text files
                f1 = open(os.path.join(d_dir, f"text.{src_lang}"))
                f2 = open(os.path.join(d_dir, f"text.{tgt_lang}"))
                src_text_lst, tgt_text_lst = zip(
                    *[(src_text.strip(), tgt_text.strip()) for src_text, tgt_text in zip(f1, f2)])

                # read gold alignment
                fn = os.path.join(d_dir, f"gold.{src_lang}-{tgt_lang}.aligned")
                util_data_dict = {"gold_alignment": [l.strip() for l in open(fn)]}

                attrs_dict = {"ds_name": ds_name, "src_lang": src_lang, "tgt_lang": tgt_lang,
                             "src_text_lst": src_text_lst, "tgt_text_lst": tgt_text_lst,
                              "util_data_dict": util_data_dict}
                yield attrs_dict

    elif args.ds_type == "OTAlign":
        data_dir = os.path.join(args.data_dir, "OTAlign")

        for d_name in os.listdir(data_dir):
            if d_name == "edinburgh":
                for portion in ["train", "dev", "test"]:
                    ds_name = f"edinburgh_{portion}"
                    if only_ds_name:
                        yield ds_name
                    else:
                        if portion != "test":
                            file_path = os.path.join(data_dir, d_name, "train.json")
                        else:
                            file_path = os.path.join(data_dir, d_name, "test.json")
                        src_text_lst, tgt_text_lst, alignment_list  = \
                            load_Edinburgh_corpus(file_path, portion=portion)
                        util_data_dict = {"gold_alignment": alignment_list}
                        attrs_dict = {"ds_name": ds_name,
                                      "src_lang": "0", "tgt_lang": "1",
                                      "src_text_lst": src_text_lst, "tgt_text_lst": tgt_text_lst,
                                      "util_data_dict": util_data_dict}
                        yield attrs_dict
            elif d_name == "msr":

                for portion in ["test"]: # TODO cannot find train and dev
                    ds_name = f"msr_{portion}"
                    if only_ds_name:
                        yield ds_name
                    else:
                        corpus_dir = os.path.join(data_dir, d_name)+"/"
                        src_text_lst, tgt_text_lst, alignment_list  = \
                            load_MSR_RTE_corpus(corpus_dir, portion=portion)
                        util_data_dict = {"gold_alignment": alignment_list}
                        attrs_dict = {"ds_name": ds_name,
                                      "src_lang": "0", "tgt_lang": "1",
                                      "src_text_lst": src_text_lst, "tgt_text_lst": tgt_text_lst,
                                      "util_data_dict": util_data_dict}
                        yield attrs_dict
            elif d_name == "mwa":
                corpus_dir = os.path.join(data_dir, d_name)
                if only_ds_name:
                    for ds_name in iter_load_MWA_corpus(corpus_dir, only_ds_name=only_ds_name):
                        yield ds_name
                else:
                    for ds_name, src_text_lst, tgt_text_lst, alignment_list in \
                            iter_load_MWA_corpus(corpus_dir):
                        util_data_dict = {"gold_alignment": alignment_list}
                        attrs_dict = {"ds_name": ds_name,
                                      "src_lang": "0", "tgt_lang": "1",
                                      "src_text_lst": src_text_lst, "tgt_text_lst": tgt_text_lst,
                                      "util_data_dict": util_data_dict}
                        yield attrs_dict

    elif args.ds_type == "hall":

        if only_ds_name:
            for ds_name in load_raw_hall(args.data_dir, only_ds_name=only_ds_name):
                yield ds_name
        else:
            ds_dict = load_raw_hall(args.data_dir, only_ds_name=only_ds_name)
            for ds_name, example_lst in ds_dict.items():
                src_text_lst, tgt_text_lst = [], []
                src_lang,  tgt_lang = example_lst[0].src_lang, example_lst[0].tgt_lang
                util_data_dict = defaultdict(list)

                for example in example_lst:
                    src_text_lst.append(example.src_text)
                    tgt_text_lst.append(example.tgt_text)
                    for k, v in example.util_data_dict.items():
                        if k not in ["src_text", "tgt_text", "src_lang",  "tgt_lang"]:
                            util_data_dict[k].append(v)

                attrs_dict = {"ds_name": ds_name,
                              "src_lang": src_lang, "tgt_lang": tgt_lang,
                              "src_text_lst": src_text_lst, "tgt_text_lst": tgt_text_lst,
                              "util_data_dict": util_data_dict}

                yield attrs_dict

    else:
        raise ValueError(f"Not supported dataset type: {args.ds_type}")

####################
### main method ####
#####################
def dump_emb(args):
    args.device = f"cuda:{args.gpu_lst.split(',')[0]}"
    # args.tokenize_style = "pmi"# "standard"
    # args.ds_type = "PMI-Align"
    # args.ds_type = "OTAlign"
    # args.ds_type = "hall"

    model_idx = args.model_idx
    
    
    args.tokenize_style = "standard" if args.ds_type == "hall" else "pmi"
    
    if args.ds_type == "OTAlign":
        args.model_name = MODEL_NAME_LST[model_idx]
    elif  args.ds_type == "PMI-Align":
        args.model_name = MODEL_NAME_LST[model_idx]
    elif  args.ds_type == "hall":
        args.tokenize_style = "pmi"
        args.model_name = MODEL_NAME_LST[model_idx]

    for attrs_dict in _iter_load_raw_ds(args):
        ds_name = attrs_dict["ds_name"]
        
        # if ds_name != 'hall_deen':
        #     continue
        
        aligner = Aligner(args, ds_name, attrs_dict["src_lang"], attrs_dict["tgt_lang"])

        print(f"Processing {ds_name} dataset")
        aligner.dump_emb(attrs_dict)

        output_file = os.path.join(args.tmp_dir, f"ds.{ds_name}.{args.model_name}.{args.tokenize_style}.pkl")
        with open(output_file, "wb") as handle:
            pickle.dump(aligner, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _set_best_align_args(args): 
    model_idx = args.model_idx
    layer_idx = args.layer_idx
    if model_idx == -1:
        layer_idx = LABSE_LAYER_IDX
        
    if args.ds_type == "OTAlign":
        args.tokenize_style = "pmi"
        # args.align_method = "pmi_align_word"
        args.merge_method = "none"
        args.use_sim_align = True
        args.layer_idx = layer_idx
        args.model_name = MODEL_NAME_LST[model_idx]


    elif args.ds_type == "PMI-Align":
        args.tokenize_style = "pmi"
        # args.align_method = "pmi_align_token"
        args.use_sim_align = False
        args.merge_method = "none"
        args.layer_idx = layer_idx
        args.model_name = MODEL_NAME_LST[model_idx]

    elif args.ds_type == "hall":
        args.tokenize_style = "pmi"
        args.align_method = "pmi_align_word"
        args.use_sim_align = False
        args.merge_method = "first"
        args.layer_idx = LABSE_LAYER_IDX 
        args.model_name = MODEL_NAME_LST[model_idx]

def run_hall_dev(args):
    # args.ds_type = "PMI-Align"
    # args.ds_type = "OTAlign"
    _set_best_align_args(args)
    args.merge_method = "first" # None
    
    result_str = ''
    
    full_results = {}
    for ds_name in _iter_load_raw_ds(args, only_ds_name=True):
        if 'test' not in ds_name and args.ds_type == "OTAlign":
            continue

        if ds_name in ['hall_deen']:
            continue
        
        print(ds_name)
        filename = os.path.join(args.tmp_dir, f"ds.{ds_name}.{args.model_name}.{args.tokenize_style}.pkl")
        with open(filename, 'rb') as fp:
            aligner: Aligner = pickle.load(fp)
        result = aligner.run_hall(args, ds_name)
        print(result)
        full_results[ds_name] = result # result['auc-ROC']
        # full_results[ds_name] = result['auc-ROC']
    
    # pkl.dump(emb_list, open("emb_list", "wb"))
    # pkl.dump(all_results, open("all_results.pkl", "wb"))
        
    key_lst = ["halomi_en:ar", "halomi_ar:en", "halomi_en:ru", "halomi_ru:en", "halomi_en:es", "halomi_es:en",
               "halomi_en:de", "halomi_de:en", "halomi_en:zh", "halomi_zh:en", "halomi_en:ks", "halomi_ks:en",
               "halomi_en:mni", "halomi_mni:en", "halomi_en:yo", "halomi_yo:en", "halomi_yo:es", "halomi_es:yo"]
    
    print("\t".join([str(round(full_results.get(k, -1), 2)) for k in key_lst]))

    print(" ".join([f"{k}" for k, v in full_results.items()]))
    print(" ".join([f"{v}" for k, v in full_results.items()]))

    
def run_align_dev(args):
    # args.ds_type = "PMI-Align"
    # args.ds_type = "OTAlign"
    _set_best_align_args(args)

    result_str = ''
    for ds_name in _iter_load_raw_ds(args, only_ds_name=True):
        if 'test' not in ds_name and args.ds_type == "OTAlign":
            continue
        
        print(ds_name)
        filename = os.path.join(args.tmp_dir, f"ds.{ds_name}.{args.model_name}.{args.tokenize_style}.pkl")
        with open(filename, 'rb') as fp:
            aligner: Aligner = pickle.load(fp)
        result = aligner.run_align(args)
        print(result)
        # result_str += ' '.join((str(result["F1 Score"]), str(result["AER"]), str(result["Precision"]), str(result["Recall"])))
        # result_str += " "
        
    print(result_str)


def main():

    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--mode",
                        default="run_hall",
                        choices=["dump_emb", "run_align", "run_hall"],
                        help="Choose which functionality to run")

    parser.add_argument(
        "--project_dir",
        type=str,
        required=True,
        help="project_dir",
    )

    parser.add_argument(
        "--gpu_lst",
        default="0,1",
        type=str,
        # required=True,
        help="gpu_lst",
    )
    
    parser.add_argument(
        "--align_method",
        default="pmi_align_token",
        type=str,

    )
    parser.add_argument(
        "--ds_type",
        default="hall",
        type=str,
        choices = ["PMI-Align", "OTAlign", "hall"]
    )
    
    parser.add_argument(
        "--error_type",
        default="hall",
        type=str,
        choices = ["hall", "omi", "entity"]
    )
    
    parser.add_argument(
        "--model_idx",
        default=-1,
        type=int,
    )
    
    parser.add_argument(
        "--layer_idx",
        default=8,
        type=int,
    )
    
    
    args = parser.parse_args()
    args.model_dir = os.path.join(args.project_dir, "models")
    args.data_dir = os.path.join(args.project_dir, "data")
    args.tmp_dir = os.path.join(args.project_dir, "tmp")

    if args.mode == "dump_emb":
        dump_emb(args)
    elif args.mode == "run_align":
        run_align_dev(args)
    elif args.mode == "run_hall":
        run_hall_dev(args)
    else:
        raise ValueError(f"Mode {args.mode} is not implemented!!!")

if __name__ == "__main__":
    main()
