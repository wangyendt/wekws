# Copyright (c) 2021 Binbin Zhang(binbzha@qq.com)
#               2022 Shaoqing Yu(954793264@qq.com)
#               2023 Jing Du(thuduj12@163.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import argparse
import copy
import logging
import os
import sys
import math
import json
from typing import Dict, Any, List, Tuple, Optional

import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader

from wekws.dataset.init_dataset import init_dataset
from wekws.model.kws_model import init_model
from wekws.utils.checkpoint import load_checkpoint
from wekws.model.loss import ctc_prefix_beam_search
from wenet.text.char_tokenizer import CharTokenizer



def get_args():
    parser = argparse.ArgumentParser(description='recognize with your model')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--test_data', required=True, help='test data file')
    parser.add_argument('--dict', default='./dict', help='dict dir')
    parser.add_argument('--gpu',
                        type=int,
                        default=-1,
                        help='gpu id for this rank, -1 for cpu')
    parser.add_argument('--checkpoint', required=True, help='checkpoint model')
    parser.add_argument('--batch_size',
                        default=16,
                        type=int,
                        help='batch size for inference')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='num of subprocess workers for reading')
    parser.add_argument('--pin_memory',
                        action='store_true',
                        default=False,
                        help='Use pinned memory buffers used for reading')
    parser.add_argument('--prefetch',
                        default=100,
                        type=int,
                        help='prefetch number')
    parser.add_argument('--score_file',
                        required=True,
                        help='output score file')
    parser.add_argument('--jit_model',
                        action='store_true',
                        default=False,
                        help='Use pinned memory buffers used for reading')
    parser.add_argument('--keywords',
                        type=str,
                        default=None,
                        help='the keywords, split with comma(,)')
    parser.add_argument('--token_file',
                        type=str,
                        default=None,
                        help='the path of tokens.txt')
    parser.add_argument('--lexicon_file',
                        type=str,
                        default=None,
                        help='the path of lexicon.txt')

    # Optional dumps for debugging / single-utterance inference.
    parser.add_argument('--dump_logits_dir',
                        type=str,
                        default=None,
                        help='If set, dump per-utt raw logits to <dir>/<key>.npy (T, V)')
    parser.add_argument('--dump_probs_dir',
                        type=str,
                        default=None,
                        help='If set, dump per-utt softmax probs to <dir>/<key>.npy (T, V)')
    parser.add_argument('--dump_decode_file',
                        type=str,
                        default=None,
                        help='If set, write per-utt jsonl with decoded outputs and labels')
    parser.add_argument('--decode_greedy',
                        action='store_true',
                        default=False,
                        help='If set, output greedy CTC decoded tokens/text')
    parser.add_argument('--decode_beam',
                        action='store_true',
                        default=False,
                        help='If set, output beam CTC decoded tokens/text (prefix beam search)')
    parser.add_argument('--decode_score_beam_size',
                        type=int,
                        default=20,
                        help='score_beam_size for beam decoding')
    parser.add_argument('--decode_path_beam_size',
                        type=int,
                        default=50,
                        help='path_beam_size for beam decoding')
    parser.add_argument('--decode_prob_threshold',
                        type=float,
                        default=0.0,
                        help='prob threshold in beam decoding (set 0.0 to disable pruning)')
    parser.add_argument('--blank_id',
                        type=int,
                        default=0,
                        help='blank token id for CTC (default: 0)')

    args = parser.parse_args()
    return args


def is_sublist(main_list, check_list):
    if len(main_list) < len(check_list):
        return -1

    if len(main_list) == len(check_list):
        return 0 if main_list == check_list else -1

    for i in range(len(main_list) - len(check_list)):
        if main_list[i] == check_list[0]:
            for j in range(len(check_list)):
                if main_list[i + j] != check_list[j]:
                    break
            else:
                return i
    else:
        return -1


def _load_id2token(dict_dir: str) -> Dict[int, str]:
    id2tok: Dict[int, str] = {}
    dict_path = os.path.join(dict_dir, 'dict.txt')
    with open(dict_path, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Format: "<token> <id>"
            arr = line.split()
            if len(arr) < 2:
                continue
            tok = " ".join(arr[:-1])
            try:
                idx = int(arr[-1])
            except ValueError:
                continue
            id2tok[idx] = tok
    return id2tok


def _ctc_greedy_decode(token_ids: List[int], blank_id: int = 0) -> List[int]:
    out: List[int] = []
    prev: Optional[int] = None
    for t in token_ids:
        if t == prev:
            continue
        prev = t
        if t == blank_id:
            continue
        out.append(t)
    return out


def _ids_to_text(ids: List[int], id2tok: Dict[int, str]) -> Tuple[str, List[str]]:
    toks = [id2tok.get(i, f'<unk:{i}>') for i in ids]
    # For this repo, labels are space-separated tokens; keep the same style.
    return (" ".join(toks)).strip(), toks


def _load_labels(test_data: str) -> Dict[str, Dict[str, Any]]:
    """Load key->metadata from jsonl data.list for pos/neg labeling and traceability."""
    out: Dict[str, Dict[str, Any]] = {}
    with open(test_data, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            key = obj.get('key')
            if not key:
                continue
            out[key] = obj
    return out


def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    test_conf = copy.deepcopy(configs['dataset_conf'])
    test_conf['filter_conf']['max_length'] = 102400
    test_conf['filter_conf']['min_length'] = 0
    test_conf['filter_conf']['token_max_length'] = 10240
    test_conf['filter_conf']['token_min_length'] = 1
    test_conf['filter_conf']['min_output_input_ratio'] = 1e-6
    test_conf['filter_conf']['max_output_input_ratio'] = 1
    test_conf['speed_perturb'] = False
    test_conf['spec_aug'] = False
    test_conf['shuffle'] = False
    feats_type = test_conf.get('feats_type', 'fbank')
    test_conf[f'{feats_type}_conf']['dither'] = 0.0
    test_conf['batch_conf']['batch_size'] = args.batch_size

    tokenizer = CharTokenizer(f'{args.dict}/dict.txt',
                              f'{args.dict}/words.txt',
                              unk='<filler>',
                              split_with_space=True)
    test_dataset = init_dataset(data_list_file=args.test_data,
                                conf=test_conf, tokenizer=tokenizer,
                                split='test')
    test_data_loader = DataLoader(test_dataset,
                                  batch_size=None,
                                  pin_memory=args.pin_memory,
                                  num_workers=args.num_workers,
                                  prefetch_factor=args.prefetch)

    if args.jit_model:
        model = torch.jit.load(args.checkpoint)
        # For script model, only cpu is supported.
        device = torch.device('cpu')
    else:
        # Init asr model from configs
        model = init_model(configs['model'])
        load_checkpoint(model, args.checkpoint)
        use_cuda = args.gpu >= 0 and torch.cuda.is_available()
        device = torch.device('cuda' if use_cuda else 'cpu')
    model = model.to(device)
    model.eval()
    score_abs_path = os.path.abspath(args.score_file)

    # 4. parse keywords tokens
    assert args.keywords is not None, 'at least one keyword is needed'
    logging.info(f"keywords is {args.keywords}, "
                 f"Chinese is converted into Unicode.")
    keywords_str = args.keywords.encode('utf-8').decode('unicode_escape')
    keywords_list = keywords_str.strip().replace(' ', '').split(',')
    keywords_list = [k for k in keywords_list if k]
    keywords_token = {}
    keywords_idxset = {0}
    keywords_strset = {'<blk>'}
    keywords_tokenmap = {'<blk>': 0}
    for keyword in keywords_list:
        strs, indexes = tokenizer.tokenize(' '.join(list(keyword)))
        indexes = tuple(indexes)
        keywords_token[keyword] = {}
        keywords_token[keyword]['token_id'] = indexes
        keywords_token[keyword]['token_str'] = ''.join('%s ' % str(i)
                                                       for i in indexes)
        [keywords_strset.add(i) for i in strs]
        [keywords_idxset.add(i) for i in indexes]
        for txt, idx in zip(strs, indexes):
            if keywords_tokenmap.get(txt, None) is None:
                keywords_tokenmap[txt] = idx

    token_print = ''
    for txt, idx in keywords_tokenmap.items():
        token_print += f'{txt}({idx}) '
    logging.info(f'Token set is: {token_print}')

    labels_by_key = _load_labels(args.test_data)
    id2tok: Optional[Dict[int, str]] = None
    if args.decode_greedy or args.decode_beam:
        id2tok = _load_id2token(args.dict)

    if args.dump_logits_dir:
        os.makedirs(args.dump_logits_dir, exist_ok=True)
    if args.dump_probs_dir:
        os.makedirs(args.dump_probs_dir, exist_ok=True)
    dump_fout = None
    if args.dump_decode_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.dump_decode_file)) or ".", exist_ok=True)
        dump_fout = open(args.dump_decode_file, 'w', encoding='utf8')

    with torch.no_grad(), open(score_abs_path, 'w', encoding='utf8') as fout:
        for batch_idx, batch_dict in enumerate(test_data_loader):
            keys = batch_dict['keys']
            feats = batch_dict['feats']
            targets = batch_dict['target'][:, 0]
            lengths = batch_dict['feats_lengths']
            label_lengths = batch_dict['target_lengths']
            feats = feats.to(device)
            lengths = lengths.to(device)
            logits_raw, _ = model(feats)  # (B, T, V)
            probs = logits_raw.softmax(2)
            probs_cpu = probs.detach().cpu()
            for i in range(len(keys)):
                key = keys[i]
                utt_len = int(lengths[i].item())

                probs_i = probs_cpu[i][:utt_len]  # (T, V)
                hyps = ctc_prefix_beam_search(probs_i, utt_len,
                                              keywords_idxset)
                hit_keyword = None
                hit_score = 1.0
                start = 0
                end = 0
                for one_hyp in hyps:
                    prefix_ids = one_hyp[0]
                    # path_score = one_hyp[1]
                    prefix_nodes = one_hyp[2]
                    assert len(prefix_ids) == len(prefix_nodes)
                    for word in keywords_token.keys():
                        lab = keywords_token[word]['token_id']
                        offset = is_sublist(prefix_ids, lab)
                        if offset != -1:
                            hit_keyword = word
                            start = prefix_nodes[offset]['frame']
                            end = prefix_nodes[offset + len(lab) - 1]['frame']
                            for idx in range(offset, offset + len(lab)):
                                hit_score *= prefix_nodes[idx]['prob']
                            break
                    if hit_keyword is not None:
                        hit_score = math.sqrt(hit_score)
                        break

                if hit_keyword is not None:
                    fout.write('{} detected {} {:.3f}\n'.format(
                        key, hit_keyword, hit_score))
                    logging.info(f"batch:{batch_idx}_{i} detect {hit_keyword} "
                                 f"in {key} from {start} to {end} frame. "
                                 f"duration {end - start}, "
                                 f"score {hit_score}, Activated.")
                else:
                    fout.write('{} rejected\n'.format(key))
                    logging.info(f"batch:{batch_idx}_{i} {key} Deactivated.")

                # Optional dumps.
                if args.dump_logits_dir:
                    logits_i = logits_raw[i][:utt_len].detach().cpu().numpy().astype(np.float32)
                    np.save(os.path.join(args.dump_logits_dir, f'{key}.npy'), logits_i)
                if args.dump_probs_dir:
                    np.save(os.path.join(args.dump_probs_dir, f'{key}.npy'),
                            probs_i.numpy().astype(np.float32))

                if dump_fout is not None:
                    meta = labels_by_key.get(key, {})
                    txt = meta.get('txt', '')
                    wav = meta.get('wav', '')
                    duration = meta.get('duration', None)
                    label_type = meta.get('label_type', None)
                    txt_nospace = (txt or '').replace(' ', '')
                    if isinstance(label_type, str) and label_type in ('positive', 'negative'):
                        is_positive = (label_type == 'positive')
                    else:
                        is_positive = any((kw in txt_nospace) for kw in keywords_list) if txt else False

                    greedy_ids: List[int] = []
                    greedy_text = ''
                    beam_ids: List[int] = []
                    beam_text = ''
                    if id2tok is not None and args.decode_greedy:
                        frame_ids = probs_i.argmax(dim=-1).tolist()
                        greedy_ids = _ctc_greedy_decode(frame_ids, blank_id=args.blank_id)
                        greedy_text, _ = _ids_to_text(greedy_ids, id2tok)
                    if id2tok is not None and args.decode_beam:
                        dec_hyps = ctc_prefix_beam_search(
                            probs_i,
                            utt_len,
                            keywords_tokenset=None,
                            score_beam_size=args.decode_score_beam_size,
                            path_beam_size=args.decode_path_beam_size,
                            prob_threshold=args.decode_prob_threshold,
                        )
                        if len(dec_hyps) > 0:
                            beam_ids = list(dec_hyps[0][0])
                            beam_text, _ = _ids_to_text(beam_ids, id2tok)

                    dump_obj = {
                        'key': key,
                        'wav': wav,
                        'txt': txt,
                        'duration': duration,
                        'label_type': label_type,
                        'is_positive': bool(is_positive),
                        'keywords': keywords_list,
                        'kws_result': {
                            'triggered': bool(hit_keyword is not None),
                            'keyword': hit_keyword,
                            'score': float(hit_score) if hit_keyword is not None else None,
                            'start_frame': int(start) if hit_keyword is not None else None,
                            'end_frame': int(end) if hit_keyword is not None else None,
                        },
                        'decode': {
                            'greedy': {
                                'token_ids': greedy_ids,
                                'text': greedy_text,
                            } if args.decode_greedy else None,
                            'beam': {
                                'token_ids': beam_ids,
                                'text': beam_text,
                                'score_beam_size': int(args.decode_score_beam_size),
                                'path_beam_size': int(args.decode_path_beam_size),
                                'prob_threshold': float(args.decode_prob_threshold),
                            } if args.decode_beam else None,
                        }
                    }
                    dump_fout.write(json.dumps(dump_obj, ensure_ascii=False) + '\n')

            if batch_idx % 10 == 0:
                print('Progress batch {}'.format(batch_idx))
                sys.stdout.flush()

    if dump_fout is not None:
        dump_fout.close()

if __name__ == '__main__':
    main()
