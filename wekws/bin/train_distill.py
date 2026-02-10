# Copyright (c) 2020 Binbin Zhang
#               2026 Wayne (feature alignment distillation)
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

"""Feature alignment distillation training for CTC-based KWS models.

Two-phase training strategy:
    Phase 1 (feature alignment): Train student backbone to match teacher's
        block-level features via MSE loss. HEAD (out_linear1 + out_linear2)
        is copied from teacher and frozen.
    Phase 2 (fine-tuning): Unfreeze HEAD with very low learning rate.
        Use MSE + CTC combined loss for end-to-end fine-tuning.

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
            --align_epochs 100 --finetune_epochs 20 \\
            --head_lr_ratio 0.01
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
        description='Feature alignment distillation training for KWS')

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
    parser.add_argument('--resume_lr', type=float, default=None,
                        help='override lr when resuming from --checkpoint '
                             '(ignore lr saved in <ckpt>.yaml)')
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

    # --- feature alignment distillation parameters ---
    parser.add_argument('--align_epochs', type=int, default=100,
                        help='number of epochs for phase 1 (feature '
                             'alignment, HEAD frozen)')
    parser.add_argument('--finetune_epochs', type=int, default=20,
                        help='number of epochs for phase 2 (MSE + CTC, '
                             'HEAD unfrozen with low lr)')
    parser.add_argument('--finetune_lr', type=float, default=None,
                        help='override backbone learning rate at the start of '
                             'phase 2 (finetune). If omitted, keep current '
                             'backbone lr from phase 1.')
    parser.add_argument('--head_lr_ratio', type=float, default=0.01,
                        help='HEAD learning rate = backbone_lr * ratio '
                             'in phase 2')
    parser.add_argument('--finetune_mse_weight_start', type=float,
                        default=0.5,
                        help='MSE weight at start of phase 2')
    parser.add_argument('--finetune_mse_weight_end', type=float, default=0.1,
                        help='MSE weight at end of phase 2')

    # --- layer mapping ---
    parser.add_argument('--layer_mapping', default='0:1,1:2,2:3',
                        help='student:teacher block mapping, '
                             'e.g. "0:1,1:2,2:3"')

    # --- lr scheduler ---
    parser.add_argument('--lr_scheduler', default='plateau',
                        choices=['plateau', 'none'],
                        help='learning rate scheduler type')
    parser.add_argument('--scheduler_start_epoch', type=int, default=-1,
                        help='start stepping scheduler from this epoch '
                             '(default: -1 means start at align_epochs)')
    parser.add_argument('--plateau_factor', type=float, default=0.5,
                        help='ReduceLROnPlateau factor')
    parser.add_argument('--plateau_patience', type=int, default=3,
                        help='ReduceLROnPlateau patience')
    parser.add_argument('--plateau_threshold', type=float, default=0.01,
                        help='ReduceLROnPlateau threshold')
    parser.add_argument('--plateau_min_lr', type=float, default=1e-6,
                        help='ReduceLROnPlateau min_lr')
    parser.add_argument('--plateau_cooldown', type=int, default=0,
                        help='ReduceLROnPlateau cooldown')

    args = parser.parse_args()
    return args


def _parse_layer_mapping(mapping_str):
    """Parse layer mapping string like '0:1,1:2,2:3' into list of tuples."""
    pairs = []
    for pair in mapping_str.split(','):
        s, t = pair.strip().split(':')
        pairs.append((int(s), int(t)))
    return pairs


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
    if torch.cuda.is_available():
        ckpt = torch.load(checkpoint_path)
    else:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
    teacher_model.load_state_dict(ckpt, strict=True)
    teacher_model = teacher_model.to(device)
    teacher_model.eval()
    for p in teacher_model.parameters():
        p.requires_grad = False
    return teacher_model, teacher_configs


def _copy_head_from_teacher(student_model, teacher_model):
    """Copy HEAD (out_linear1 + out_linear2) from teacher to student.

    Requires that student and teacher have matching dimensions:
    - out_linear1: (linear_dim -> output_affine_dim) must be identical
    - out_linear2: (output_affine_dim -> output_dim) must be identical
    """
    # Get the raw model (unwrap DDP if needed)
    s_backbone = student_model.module.backbone \
        if hasattr(student_model, 'module') else student_model.backbone
    t_backbone = teacher_model.module.backbone \
        if hasattr(teacher_model, 'module') else teacher_model.backbone

    # Copy out_linear1
    s_ol1 = s_backbone.out_linear1.state_dict()
    t_ol1 = t_backbone.out_linear1.state_dict()
    for key in s_ol1:
        if s_ol1[key].shape != t_ol1[key].shape:
            raise ValueError(
                f'out_linear1.{key} shape mismatch: '
                f'student {s_ol1[key].shape} vs teacher {t_ol1[key].shape}. '
                f'Ensure student output_affine_dim matches teacher.')
    s_backbone.out_linear1.load_state_dict(t_ol1)
    logging.info('Copied out_linear1 from teacher to student')

    # Copy out_linear2
    s_ol2 = s_backbone.out_linear2.state_dict()
    t_ol2 = t_backbone.out_linear2.state_dict()
    for key in s_ol2:
        if s_ol2[key].shape != t_ol2[key].shape:
            raise ValueError(
                f'out_linear2.{key} shape mismatch: '
                f'student {s_ol2[key].shape} vs teacher {t_ol2[key].shape}. '
                f'Ensure student num_keywords matches teacher output_dim.')
    s_backbone.out_linear2.load_state_dict(t_ol2)
    logging.info('Copied out_linear2 from teacher to student')


def _freeze_head(model):
    """Freeze HEAD parameters (out_linear1 + out_linear2)."""
    backbone = model.module.backbone \
        if hasattr(model, 'module') else model.backbone
    frozen_count = 0
    for name, param in backbone.named_parameters():
        if 'out_linear1' in name or 'out_linear2' in name:
            param.requires_grad = False
            frozen_count += param.numel()
    logging.info('Frozen HEAD: %d parameters', frozen_count)
    return frozen_count


def _unfreeze_head(model):
    """Unfreeze HEAD parameters (out_linear1 + out_linear2)."""
    backbone = model.module.backbone \
        if hasattr(model, 'module') else model.backbone
    unfrozen_count = 0
    for name, param in backbone.named_parameters():
        if 'out_linear1' in name or 'out_linear2' in name:
            param.requires_grad = True
            unfrozen_count += param.numel()
    logging.info('Unfrozen HEAD: %d parameters', unfrozen_count)
    return unfrozen_count


def _get_param_groups(model, base_lr, head_lr_ratio):
    """Create parameter groups with different learning rates.

    Returns:
        list of dicts for optimizer: backbone params at base_lr,
        HEAD params at base_lr * head_lr_ratio.
    """
    backbone_mod = model.module.backbone \
        if hasattr(model, 'module') else model.backbone

    head_param_ids = set()
    head_params = []
    for name, param in backbone_mod.named_parameters():
        if 'out_linear1' in name or 'out_linear2' in name:
            head_param_ids.add(id(param))
            head_params.append(param)

    backbone_params = []
    for param in model.parameters():
        if id(param) not in head_param_ids and param.requires_grad:
            backbone_params.append(param)

    param_groups = [
        {'params': backbone_params, 'lr': base_lr, 'name': 'backbone'},
        {'params': head_params, 'lr': base_lr * head_lr_ratio, 'name': 'head'},
    ]
    return param_groups


def _init_student_from_teacher(student_model, teacher_model):
    """Initialize student backbone parameters from teacher by slicing.

    For each parameter in the student:
      - If the same key exists in the teacher with identical shape, copy it.
      - If the student shape is element-wise <= teacher shape (same ndim),
        slice the first N elements along each dimension.
      - Otherwise skip (keep random init).

    Note: HEAD (out_linear1 + out_linear2) is handled separately by
    _copy_head_from_teacher(), so this function focuses on the backbone
    (in_linear, fsmn blocks).
    """
    s_state = student_model.state_dict()
    t_state = teacher_model.state_dict()
    initialized = []
    skipped = []
    for key in s_state:
        # Skip HEAD - it will be copied exactly by _copy_head_from_teacher
        if 'out_linear1' in key or 'out_linear2' in key:
            skipped.append('{} (HEAD, handled separately)'.format(key))
            continue
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
                '{} (sliced {}>{})'.format(key, list(t_shape), list(s_shape)))
        else:
            skipped.append(
                '{} (shape mismatch: teacher {} vs student {})'.format(
                    key, list(t_shape), list(s_shape)))
    student_model.load_state_dict(s_state)
    logging.info('Init backbone from teacher: initialized %d/%d params',
                 len(initialized), len(s_state))
    for item in initialized:
        logging.info('  [INIT] %s', item)
    if skipped:
        logging.info('Init backbone from teacher: skipped %d params',
                     len(skipped))
        for item in skipped:
            logging.info('  [SKIP] %s', item)


def _compute_mse_weight(epoch, align_epochs, finetune_epochs,
                        mse_start, mse_end):
    """Compute MSE weight for the finetune phase (linear decay).

    Returns mse_start at start of phase 2, linearly decaying to mse_end.
    """
    if finetune_epochs <= 1:
        return mse_start
    progress = (epoch - align_epochs) / (finetune_epochs - 1)
    progress = min(max(progress, 0.0), 1.0)
    return mse_start + (mse_end - mse_start) * progress


def _build_scheduler(args, optimizer):
    if args.lr_scheduler == 'none':
        return None
    if args.lr_scheduler == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=args.plateau_factor,
            patience=args.plateau_patience,
            min_lr=args.plateau_min_lr,
            threshold=args.plateau_threshold,
            cooldown=args.plateau_cooldown,
        )
    raise ValueError(f'Unsupported lr_scheduler: {args.lr_scheduler}')


def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    set_mannul_seed(args.seed)
    print(args)

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    layer_mapping = _parse_layer_mapping(args.layer_mapping)
    total_epochs = args.align_epochs + args.finetune_epochs

    scheduler_start_epoch = args.scheduler_start_epoch
    if scheduler_start_epoch < 0:
        # Default behavior: keep lr constant during alignment, start scheduling
        # at finetune phase.
        scheduler_start_epoch = args.align_epochs

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

    # ---- Device placement ----
    use_cuda = gpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # ---- Load teacher model (frozen) ----
    teacher_config_path = _resolve_teacher_config(args)
    logging.info('Loading teacher from %s (config: %s)',
                 args.teacher_checkpoint, teacher_config_path)
    teacher_model, teacher_configs = _load_teacher(
        teacher_config_path, args.teacher_checkpoint, device)
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    teacher_output_dim = teacher_configs['model']['output_dim']
    print('Teacher Model Parameters: {:,} (frozen)'.format(teacher_params))
    print('Teacher output_dim: {}'.format(teacher_output_dim))

    # Verify num_keywords matches teacher
    if args.num_keywords != teacher_output_dim:
        logging.warning(
            'num_keywords (%d) != teacher output_dim (%d). '
            'Overriding num_keywords to match teacher.',
            args.num_keywords, teacher_output_dim)
        args.num_keywords = teacher_output_dim

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

    # Override max_epoch from training config to total_epochs
    configs['training_config']['max_epoch'] = total_epochs

    student_model = init_model(configs['model'])

    if rank == 0:
        saved_config_path = os.path.join(args.model_dir, 'config.yaml')
        os.makedirs(args.model_dir, exist_ok=True)
        with open(saved_config_path, 'w') as fout:
            fout.write(yaml.dump(configs))
        print(student_model)

    # ---- Initialize student from teacher ----
    # Step 1: Initialize backbone (in_linear, fsmn blocks) by slicing
    logging.info('Initializing student backbone from teacher (slicing)...')
    _init_student_from_teacher(student_model, teacher_model)

    # Step 2: Copy HEAD exactly from teacher
    logging.info('Copying HEAD (out_linear1 + out_linear2) from teacher...')
    _copy_head_from_teacher(student_model, teacher_model)

    # Print parameter counts
    param_details = count_parameters_detailed(student_model)
    print('=' * 60)
    print('Student Model Parameters:')
    print('  - Backbone params:  {:,}'.format(param_details['backbone']))
    print('  - Head params:      {:,}'.format(param_details['head']))
    print('  - Total params:     {:,}'.format(param_details['total']))
    print('=' * 60)

    # ---- Freeze HEAD for Phase 1 ----
    _freeze_head(student_model)

    # ---- Load student checkpoint (resume) ----
    executor = DistillExecutor()
    if args.checkpoint is not None:
        infos = load_checkpoint(student_model, args.checkpoint, strict=True)
    else:
        infos = {}
    start_epoch = infos.get('epoch', -1) + 1
    cv_loss = infos.get('cv_loss', 0.0)
    if args.resume_lr is not None:
        configs['optim_conf']['lr'] = args.resume_lr
    else:
        lr_last_epoch = infos.get('lr', configs['optim_conf']['lr'])
        configs['optim_conf']['lr'] = lr_last_epoch

    # ---- DDP for student only ----
    if world_size > 1:
        assert torch.cuda.is_available()
        student_model.cuda()
        student_model = torch.nn.parallel.DistributedDataParallel(
            student_model, find_unused_parameters=True)
    else:
        student_model = student_model.to(device)

    # ---- Phase 1 optimizer (backbone only, HEAD frozen) ----
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, student_model.parameters()),
        **configs['optim_conf'])
    scheduler = _build_scheduler(args, optimizer)

    training_config = configs['training_config']
    training_config['min_duration'] = args.min_duration
    training_config['layer_mapping'] = layer_mapping
    training_config['scheduler_start_epoch'] = scheduler_start_epoch
    training_config['lr_scheduler'] = args.lr_scheduler

    # ---- Print distillation config ----
    if rank == 0:
        print('=' * 60)
        print('Feature Alignment Distillation Config:')
        print('  - Phase 1 (align):     {} epochs (0 ~ {})'.format(
            args.align_epochs, args.align_epochs - 1))
        print('  - Phase 2 (finetune):  {} epochs ({} ~ {})'.format(
            args.finetune_epochs, args.align_epochs, total_epochs - 1))
        print('  - Finetune backbone lr: {}'.format(
            args.finetune_lr if args.finetune_lr is not None else 'keep'))
        print('  - HEAD lr ratio:       {}'.format(args.head_lr_ratio))
        print('  - Finetune MSE weight: {} -> {}'.format(
            args.finetune_mse_weight_start, args.finetune_mse_weight_end))
        print('  - Layer mapping:       {}'.format(layer_mapping))
        print('  - LR scheduler:        {} (start_epoch={})'.format(
            args.lr_scheduler, scheduler_start_epoch))
        print('  - Total epochs:        {}'.format(total_epochs))
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
    phase2_optimizer_created = False

    for epoch in range(start_epoch, total_epochs):
        training_config['epoch'] = epoch

        # Determine phase
        if epoch < args.align_epochs:
            phase = 'align'
            mse_weight = 1.0
        else:
            phase = 'finetune'
            mse_weight = _compute_mse_weight(
                epoch, args.align_epochs, args.finetune_epochs,
                args.finetune_mse_weight_start,
                args.finetune_mse_weight_end)

            # Transition to Phase 2: unfreeze HEAD, rebuild optimizer
            if not phase2_optimizer_created:
                logging.info('=' * 60)
                logging.info('PHASE TRANSITION: align -> finetune at epoch %d',
                             epoch)
                logging.info('Unfreezing HEAD with lr ratio %.4f',
                             args.head_lr_ratio)
                logging.info('=' * 60)

                _unfreeze_head(student_model)

                # Get current backbone lr from phase 1 optimizer
                current_lr = optimizer.param_groups[0]['lr']
                base_lr = current_lr
                if args.finetune_lr is not None:
                    base_lr = args.finetune_lr
                    logging.info('Override finetune backbone lr: %.6f '
                                 '(was %.6f)', base_lr, current_lr)
                else:
                    logging.info('Current backbone lr: %.6f', current_lr)
                logging.info('HEAD lr will be: %.6f',
                             base_lr * args.head_lr_ratio)

                # Create new optimizer with param groups
                param_groups = _get_param_groups(
                    student_model, base_lr, args.head_lr_ratio)
                # Preserve weight_decay from config
                wd = configs['optim_conf'].get('weight_decay', 0.0001)
                optimizer = optim.Adam(param_groups, weight_decay=wd)
                scheduler = _build_scheduler(args, optimizer)
                phase2_optimizer_created = True

        training_config['phase'] = phase
        training_config['mse_weight'] = mse_weight

        # Log epoch info
        lr_backbone = optimizer.param_groups[0]['lr']
        lr_head = (optimizer.param_groups[1]['lr']
                   if len(optimizer.param_groups) > 1 else 0.0)
        logging.info(
            'Epoch %d  [%s]  lr_backbone=%.6f  lr_head=%.6f  '
            'mse_weight=%.2f',
            epoch, phase, lr_backbone, lr_head, mse_weight)

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
                'lr': lr_backbone,
                'cv_loss': cv_loss,
                'phase': phase,
                'mse_weight': mse_weight,
            })
            if writer is not None:
                writer.add_scalar('epoch/cv_loss', cv_loss, epoch)
                writer.add_scalar('epoch/cv_acc', cv_acc, epoch)
                writer.add_scalar('epoch/lr_backbone', lr_backbone, epoch)
                writer.add_scalar('epoch/lr_head', lr_head, epoch)
                writer.add_scalar('epoch/mse_weight', mse_weight, epoch)

        final_epoch = epoch
        if scheduler is not None and epoch >= scheduler_start_epoch:
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
