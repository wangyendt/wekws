#!/usr/bin/env python3
# Copyright (c) 2021 Jingyong Hou (houjingyong@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import numpy as np
import random


def set_mannul_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters_detailed(model):
    """
    统计模型参数量的详细信息，分别统计 backbone 和 head（out_linear2）的参数量
    
    Returns:
        dict: 包含 'backbone', 'head', 'total' 三个键的字典
    """
    total_params = 0
    backbone_params = 0
    head_params = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count = param.numel()
            total_params += param_count
            
            # 判断是否属于输出层（head）
            # out_linear2 是最终的输出层
            if 'out_linear2' in name:
                head_params += param_count
            else:
                backbone_params += param_count
    
    return {
        'backbone': backbone_params,
        'head': head_params,
        'total': total_params
    }
