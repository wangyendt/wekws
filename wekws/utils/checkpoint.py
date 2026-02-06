# Copyright (c) 2021 Binbin Zhang
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

import logging
import os
import re
from typing import Dict, Optional

import yaml
import torch


def _parse_dict(dict_path: str) -> Dict[str, int]:
    """Parse dict.txt and return {token: id} mapping (only id >= 0)."""
    tok2id: Dict[str, int] = {}
    with open(dict_path, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            tok = parts[0]
            try:
                idx = int(parts[-1])
            except ValueError:
                continue
            if idx >= 0:
                tok2id[tok] = idx
    return tok2id


def _build_id_map(old_dict_path: str,
                  new_dict_path: str) -> Dict[int, int]:
    """Build {new_id: old_id} mapping by matching token strings."""
    old_tok2id = _parse_dict(old_dict_path)
    new_tok2id = _parse_dict(new_dict_path)
    id_map: Dict[int, int] = {}
    for tok, new_id in new_tok2id.items():
        if tok in old_tok2id:
            id_map[new_id] = old_tok2id[tok]
    return id_map


def load_checkpoint(model: torch.nn.Module,
                    path: str,
                    strict: bool = True,
                    old_dict_dir: Optional[str] = None,
                    new_dict_dir: Optional[str] = None) -> dict:
    if torch.cuda.is_available():
        logging.info('Checkpoint: loading from checkpoint %s for GPU' % path)
        checkpoint = torch.load(path)
    else:
        logging.info('Checkpoint: loading from checkpoint %s for CPU' % path)
        checkpoint = torch.load(path, map_location='cpu')
    if not strict:
        model_state = model.state_dict()
        filtered = {}
        mismatched = []
        for k, v in checkpoint.items():
            if k in model_state and model_state[k].shape == v.shape:
                filtered[k] = v
            else:
                mismatched.append(k)

        # Weight surgery: copy matching rows from pretrained output layer
        if (mismatched and old_dict_dir and new_dict_dir
                and old_dict_dir != new_dict_dir):
            old_dict_path = os.path.join(old_dict_dir, 'dict.txt')
            new_dict_path = os.path.join(new_dict_dir, 'dict.txt')
            if (os.path.exists(old_dict_path)
                    and os.path.exists(new_dict_path)):
                id_map = _build_id_map(old_dict_path, new_dict_path)
                if id_map:
                    surgery_keys = [k for k in mismatched
                                    if 'out_linear2' in k]
                    for k in surgery_keys:
                        old_param = checkpoint[k]
                        new_param = model_state[k].clone()
                        copied = 0
                        for new_id, old_id in id_map.items():
                            if (old_id < old_param.shape[0]
                                    and new_id < new_param.shape[0]):
                                new_param[new_id] = old_param[old_id]
                                copied += 1
                        filtered[k] = new_param
                        mismatched.remove(k)
                        logging.info(
                            'Weight surgery: %s, copied %d/%d rows '
                            'from pretrained checkpoint', k, copied,
                            new_param.shape[0])

        if mismatched:
            logging.warning('Checkpoint: skipped mismatched keys: %s',
                            mismatched)
        checkpoint = filtered
    result = model.load_state_dict(checkpoint, strict=strict)
    if not strict and (result.missing_keys or result.unexpected_keys):
        logging.warning('Checkpoint: missing keys: %s', result.missing_keys)
        logging.warning('Checkpoint: unexpected keys: %s',
                        result.unexpected_keys)
    info_path = re.sub('.pt$', '.yaml', path)
    configs = {}
    if os.path.exists(info_path):
        with open(info_path, 'r') as fin:
            configs = yaml.load(fin, Loader=yaml.FullLoader)
    return configs


def save_checkpoint(model: torch.nn.Module, path: str, infos=None):
    '''
    Args:
        infos (dict or None): any info you want to save.
    '''
    logging.info('Checkpoint: save to checkpoint %s' % path)
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(state_dict, path)
    info_path = re.sub('.pt$', '.yaml', path)
    if infos is None:
        infos = {}
    with open(info_path, 'w') as fout:
        data = yaml.dump(infos)
        fout.write(data)
