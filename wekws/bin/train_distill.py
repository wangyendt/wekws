# Copyright (c) 2020 Binbin Zhang
#               2026 Wayne (knowledge distillation)
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

"""Knowledge distillation training for CTC-based KWS models.

Usage (launched via torch.distributed.run):
    python -m torch.distributed.run --standalone --nproc_per_node=4 \\
        wekws/bin/train_distill.py \\
            --config conf/fsmn_ctc_student_mini.yaml \\
            --teacher_checkpoint exp/fsmn_ctc_top20_weight_surgery/79.pt \\
            --train_data data/train/data.list \\
            --cv_data   data/dev/data.list \\
            --model_dir exp/fsmn_ctc_distill_mini \\
            --num_keywords 20 \\
            --dict dict_top20 \\
            --gpus 0,1,2,3 \\
            --cmvn_file data/global_cmvn.kaldi --norm_var \\
            --kd_temperature 2.0 \\
            --kd_lambda_init 0.7 --kd_lambda_final 0.5 \\
            --kd_lambda_switch_epoch 20 --finetune_epochs 10
"""

from __future__ import print_function

import argparse
import copy
import logging
import os

import torch
import torch.distributed as dist
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from wekws.dataset.init_dataset import init_dataset
from wekws.utils.checkpoint import load_checkpoint, save_checkpoint
from wekws.model.kws_model import init_model
from wekws.utils.executor_distill import DistillExecutor
from wekws.utils.train_utils import count_parameters, set_mannul_seed, \
    count_parameters_detailed
from wenet.text.char_tokenizer import CharTokenizer


def get_args():
    parser = argparse.ArgumentParser(
        description='Knowledge distillation training for KWS')

    # --- student model ---
    parser.add_argument('--config', required=True,
                        help='student model config file')
    parser.add_argument('--train_data', required=True, help='train data file')
    parser.add_argument('--cv_data', required=True, help='cv data file')
    parser.add_argument('--model_dir', required=True, help='save model dir')
    parser.add_argument('--gpus', default='-1',
                        help='gpu list, separated with ","')
    parser.add_argument('--seed', type=int, default=777, help='random seed')
    parser.add_argument('--checkpoint', default=None,
                        help='student checkpoint to resume from')
    parser.add_argument('--num_workers', default=0, type=int,
                        help='number of subprocess workers for reading')
    parser.add_argument('--pin_memory', action='store_true', default=False,
                        help='pin memory for DataLoader')
    parser.add_argument('--cmvn_file', default=None, help='global cmvn file')
    parser.add_argument('--norm_var', action='store_true', default=False,
                        help='normalize variance in CMVN')
    parser.add_argument('--dict', default='./dict', help='dict dir')
    parser.add_argument('--num_keywords', default=20, type=int,
                        help='number of keywords (output dim)')
    parser.add_argument('--min_duration', default=50, type=int,
                        help='min duration frames of the keyword')
    parser.add_argument('--prefetch', default=100, type=int,
                        help='prefetch number')
    parser.add_argument('--ddp.dist_backend', dest='dist_backend',
                        default='nccl', choices=['nccl', 'gloo'],
                        help='distributed backend')
    parser.add_argument('--tensorboard_dir', default='tensorboard',
                        help='tensorboard log dir')

    # --- teacher model ---
    parser.add_argument('--teacher_checkpoint', required=True,
                        help='path to teacher .pt checkpoint')
    parser.add_argument('--teacher_config', default=None,
                        help='teacher config.yaml (auto-derived from '
                             'teacher_checkpoint dir if omitted)')

    # --- distillation hyper-parameters ---
    parser.add_argument('--kd_temperature', type=float, default=2.0,
                        help='softmax temperature T for KD')
    parser.add_argument('--kd_lambda_init', type=float, default=0.7,
                        help='lambda for early epochs (CTC weight)')
    parser.add_argument('--kd_lambda_final', type=float, default=0.5,
                        help='lambda for mid-training (CTC weight)')
    parser.add_argument('--kd_lambda_switch_epoch', type=int, default=20,
                        help='epoch to switch from lambda_init to '
                             'lambda_final')
    parser.add_argument('--finetune_epochs', type=int, default=10,
                        help='last N epochs use pure CTC (lambda=1.0)')

    # --- teacher weight initialization ---
    parser.add_argument('--init_from_teacher',
                        default=False,
                        type=lambda x: str(x).lower() == 'true',
                        help='initialize student backbone from teacher '
                             'weights by slicing (true/false)')

    args = parser.parse_args()
    return args


def _resolve_teacher_config(args):
    """Resolve teacher config path from checkpoint path if not specified."""
    if args.teacher_config is not None:
        return args.teacher_config
    teacher_dir = os.path.dirname(args.teacher_checkpoint)
    candidate = os.path.join(teacher_dir, 'config.yaml')
    if os.path.exists(candidate):
        return candidate
    raise FileNotFoundError(
        f'Cannot find teacher config.yaml at {candidate}. '
        'Please specify --teacher_config explicitly.')


def _load_teacher(config_path, checkpoint_path, device):
    """Build and load teacher model (frozen, eval mode)."""
    with open(config_path, 'r') as f:
        teacher_configs = yaml.load(f, Loader=yaml.FullLoader)

    teacher_model = init_model(teacher_configs['model'])
    # Load weights (strict=True, no weight surgery needed)
    if torch.cuda.is_available():
        ckpt = torch.load(checkpoint_path)
    else:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
    teacher_model.load_state_dict(ckpt, strict=True)
    teacher_model = teacher_model.to(device)
    teacher_model.eval()
    # Freeze all parameters
    for p in teacher_model.parameters():
        p.requires_grad = False
    return teacher_model


def _init_student_from_teacher(student_model, teacher_model):
    """Initialize student parameters from teacher by slicing.

    For each parameter in the student:
      - If the same key exists in the teacher with identical shape, copy it.
      - If the student shape is element-wise <= teacher shape (same ndim),
        slice the first N elements along each dimension.
      - Otherwise skip (keep random init).

    This enables initializing a smaller FSMN student from a larger teacher
    even when layer dimensions differ (e.g. 250->160, 128->64, 140->96).
    The student's 3 FSMN blocks are initialized from the teacher's first 3
    blocks (teacher block 3 is automatically skipped since the key doesn't
    exist in the student).
    """
    s_state = student_model.state_dict()
    t_state = teacher_model.state_dict()
    initialized = []
    skipped = []
    for key in s_state:
        if key not in t_state:
            skipped.append(key)
            continue
        s_shape = s_state[key].shape
        t_shape = t_state[key].shape
        if s_shape == t_shape:
            s_state[key] = t_state[key].clone()
            initialized.append(key)
        elif (len(s_shape) == len(t_shape)
              and all(s <= t for s, t in zip(s_shape, t_shape))):
            slices = tuple(slice(0, s) for s in s_shape)
            s_state[key] = t_state[key][slices].clone()
            initialized.append(
                '{} (sliced {}→{})'.format(key, list(t_shape), list(s_shape)))
        else:
            skipped.append(
                '{} (shape mismatch: teacher {} vs student {})'.format(
                    key, list(t_shape), list(s_shape)))
    student_model.load_state_dict(s_state)
    logging.info('Init from teacher: initialized %d/%d params',
                 len(initialized), len(s_state))
    for item in initialized:
        logging.info('  [INIT] %s', item)
    if skipped:
        logging.info('Init from teacher: skipped %d params', len(skipped))
        for item in skipped:
            logging.info('  [SKIP] %s', item)


def _compute_lambda(epoch, max_epoch, args):
    """Compute the current lambda (CTC weight) based on epoch schedule.

    Schedule:
        epoch < kd_lambda_switch_epoch:  lambda_init  (default 0.7)
        kd_lambda_switch_epoch <= epoch < max_epoch - finetune_epochs:
                                         lambda_final (default 0.5)
        epoch >= max_epoch - finetune_epochs:  1.0 (pure CTC)
    """
    finetune_start = max_epoch - args.finetune_epochs
    if epoch >= finetune_start:
        return 1.0
    if epoch < args.kd_lambda_switch_epoch:
        return args.kd_lambda_init
    return args.kd_lambda_final


def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    set_mannul_seed(args.seed)
    print(args)

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    # ---- Distributed setup ----
    rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    gpu = int(args.gpus.split(',')[rank])
    torch.cuda.set_device(gpu)
    if world_size > 1:
        logging.info('Distill training on multiple gpus, this gpu %d', gpu)
        dist.init_process_group(backend=args.dist_backend)

    # ---- Datasets ----
    train_conf = configs['dataset_conf']
    cv_conf = copy.deepcopy(train_conf)
    cv_conf['speed_perturb'] = False
    cv_conf['spec_aug'] = False
    cv_conf['shuffle'] = False

    tokenizer = CharTokenizer(f'{args.dict}/dict.txt',
                              f'{args.dict}/words.txt',
                              unk='<filler>',
                              split_with_space=True)
    train_dataset = init_dataset(data_list_file=args.train_data,
                                 conf=train_conf, tokenizer=tokenizer)
    cv_dataset = init_dataset(data_list_file=args.cv_data, conf=cv_conf,
                              tokenizer=tokenizer, split='dev')
    train_data_loader = DataLoader(train_dataset, batch_size=None,
                                   pin_memory=args.pin_memory,
                                   num_workers=args.num_workers,
                                   prefetch_factor=args.prefetch)
    cv_data_loader = DataLoader(cv_dataset, batch_size=None,
                                pin_memory=args.pin_memory,
                                num_workers=args.num_workers,
                                prefetch_factor=args.prefetch)

    # ---- Build student model ----
    feats_type = train_conf.get('feats_type', 'fbank')
    input_dim = train_conf[f'{feats_type}_conf']['num_mel_bins']
    output_dim = args.num_keywords

    if 'input_dim' not in configs['model']:
        configs['model']['input_dim'] = input_dim
    configs['model']['output_dim'] = output_dim
    if args.cmvn_file is not None:
        configs['model']['cmvn'] = {}
        configs['model']['cmvn']['norm_var'] = args.norm_var
        configs['model']['cmvn']['cmvn_file'] = args.cmvn_file

    student_model = init_model(configs['model'])

    if rank == 0:
        saved_config_path = os.path.join(args.model_dir, 'config.yaml')
        os.makedirs(args.model_dir, exist_ok=True)
        with open(saved_config_path, 'w') as fout:
            fout.write(yaml.dump(configs))
        print(student_model)

    # Print parameter counts
    param_details = count_parameters_detailed(student_model)
    print('=' * 60)
    print('Student Model Parameters:')
    print('  - Backbone params:  {:,}'.format(param_details['backbone']))
    print('  - Head params:      {:,}'.format(param_details['head']))
    print('  - Total params:     {:,}'.format(param_details['total']))
    print('=' * 60)

    # ---- Load student checkpoint (resume) ----
    executor = DistillExecutor()
    if args.checkpoint is not None:
        infos = load_checkpoint(student_model, args.checkpoint, strict=True)
    else:
        infos = {}
    start_epoch = infos.get('epoch', -1) + 1
    cv_loss = infos.get('cv_loss', 0.0)
    lr_last_epoch = infos.get('lr', configs['optim_conf']['lr'])
    configs['optim_conf']['lr'] = lr_last_epoch

    # ---- Device placement ----
    use_cuda = gpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # ---- Load teacher model (frozen) ----
    teacher_config_path = _resolve_teacher_config(args)
    logging.info('Loading teacher from %s (config: %s)',
                 args.teacher_checkpoint, teacher_config_path)
    teacher_model = _load_teacher(teacher_config_path,
                                  args.teacher_checkpoint, device)
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    print('Teacher Model Parameters: {:,} (frozen)'.format(teacher_params))

    # ---- Init student from teacher weights (optional) ----
    if args.init_from_teacher:
        logging.info('Initializing student from teacher weights (slicing)...')
        _init_student_from_teacher(student_model, teacher_model)

    # ---- DDP for student only ----
    if world_size > 1:
        assert torch.cuda.is_available()
        student_model.cuda()
        student_model = torch.nn.parallel.DistributedDataParallel(
            student_model)
    else:
        student_model = student_model.to(device)

    # ---- Optimizer & scheduler (student only) ----
    optimizer = optim.Adam(student_model.parameters(),
                           **configs['optim_conf'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3,
        min_lr=1e-6, threshold=0.01)

    training_config = configs['training_config']
    training_config['min_duration'] = args.min_duration
    num_epochs = training_config.get('max_epoch', 100)

    # ---- Print distillation config ----
    if rank == 0:
        print('=' * 60)
        print('Distillation Config:')
        print('  - Temperature:         {}'.format(args.kd_temperature))
        print('  - Lambda init:         {}'.format(args.kd_lambda_init))
        print('  - Lambda final:        {}'.format(args.kd_lambda_final))
        print('  - Lambda switch epoch: {}'.format(
            args.kd_lambda_switch_epoch))
        print('  - Finetune epochs:     {} (pure CTC from epoch {})'.format(
            args.finetune_epochs, num_epochs - args.finetune_epochs))
        print('  - Init from teacher:   {}'.format(args.init_from_teacher))
        print('  - Total epochs:        {}'.format(num_epochs))
        print('=' * 60)

    model_dir = args.model_dir
    writer = None
    if rank == 0:
        os.makedirs(model_dir, exist_ok=True)
        exp_id = os.path.basename(model_dir)
        writer = SummaryWriter(os.path.join(args.tensorboard_dir, exp_id))

    if start_epoch == 0 and rank == 0:
        save_model_path = os.path.join(model_dir, 'init.pt')
        save_checkpoint(student_model, save_model_path)

    # ---- Training loop ----
    final_epoch = None
    for epoch in range(start_epoch, num_epochs):
        training_config['epoch'] = epoch
        lam = _compute_lambda(epoch, num_epochs, args)
        training_config['kd_temperature'] = args.kd_temperature
        training_config['kd_lambda'] = lam

        lr = optimizer.param_groups[0]['lr']
        logging.info('Epoch %d  TRAIN  lr=%.6f  lambda=%.2f  T=%.1f%s',
                     epoch, lr, lam, args.kd_temperature,
                     '  [pure CTC]' if lam >= 1.0 else '')

        executor.train(teacher_model, student_model, optimizer,
                       train_data_loader, device, writer, training_config)

        cv_loss, cv_acc = executor.cv(
            student_model, cv_data_loader, device, training_config,
            teacher_model=teacher_model)
        logging.info('Epoch %d  CV  cv_loss=%.6f  cv_acc=%.6f',
                     epoch, cv_loss, cv_acc)

        if rank == 0:
            save_model_path = os.path.join(model_dir,
                                           '{}.pt'.format(epoch))
            save_checkpoint(student_model, save_model_path, {
                'epoch': epoch,
                'lr': lr,
                'cv_loss': cv_loss,
                'kd_lambda': lam,
            })
            if writer is not None:
                writer.add_scalar('epoch/cv_loss', cv_loss, epoch)
                writer.add_scalar('epoch/cv_acc', cv_acc, epoch)
                writer.add_scalar('epoch/lr', lr, epoch)
                writer.add_scalar('epoch/kd_lambda', lam, epoch)

        final_epoch = epoch
        scheduler.step(cv_loss)

    if final_epoch is not None and rank == 0:
        final_model_path = os.path.join(model_dir, 'final.pt')
        if os.path.exists(final_model_path):
            os.remove(final_model_path)
        os.symlink('{}.pt'.format(final_epoch), final_model_path)
        if writer is not None:
            writer.close()


if __name__ == '__main__':
    main()
