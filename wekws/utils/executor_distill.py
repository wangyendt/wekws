# Copyright (c) 2021 Binbin Zhang
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

"""Feature alignment distillation executor for CTC-based KWS models.

This module provides DistillExecutor which trains a student model using
block-level feature alignment (MSE loss) against a frozen teacher model.

Two-phase training:
    Phase 1 (feature alignment): Only MSE loss on block outputs, HEAD frozen.
    Phase 2 (fine-tuning):       alpha * MSE + (1-alpha) * CTC, HEAD unfrozen.

Block mapping (student 3 blocks -> teacher 4 blocks):
    student block 0 -> teacher block 1
    student block 1 -> teacher block 2
    student block 2 -> teacher block 3
"""

import logging

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from wekws.model.loss import criterion


def _unwrap_model(model):
    """Unwrap DistributedDataParallel to access the underlying module."""
    return model.module if hasattr(model, 'module') else model


class DistillExecutor:

    def __init__(self):
        self.step = 0

    @staticmethod
    def _make_length_mask(lengths: torch.Tensor,
                          max_len: int) -> torch.Tensor:
        """Create a boolean mask from sequence lengths.

        Args:
            lengths: (B,) integer tensor of valid lengths.
            max_len: maximum sequence length T.

        Returns:
            mask: (B, T, 1) float tensor, 1.0 for valid, 0.0 for padding.
        """
        mask = (torch.arange(max_len, device=lengths.device).unsqueeze(0)
                < lengths.unsqueeze(1))
        return mask.unsqueeze(-1).float()

    @staticmethod
    def block_mse_loss(student_block_outputs, teacher_block_outputs,
                       layer_mapping, mask):
        """Compute averaged MSE loss across mapped block pairs.

        Args:
            student_block_outputs: list of (B, T, D_s) tensors from student.
            teacher_block_outputs: list of (B, T, D_t) tensors from teacher.
            layer_mapping: list of (student_idx, teacher_idx) tuples.
            mask: (B, T, 1) float mask for valid frames.

        Returns:
            Scalar MSE loss averaged over all mapped pairs and valid frames.
        """
        if len(student_block_outputs) == 0 or len(teacher_block_outputs) == 0:
            raise ValueError(
                'Empty block outputs: '
                f'student={len(student_block_outputs)}, '
                f'teacher={len(teacher_block_outputs)}')
        if len(layer_mapping) == 0:
            raise ValueError('layer_mapping is empty')
        max_s = max(s_idx for s_idx, _ in layer_mapping)
        max_t = max(t_idx for _, t_idx in layer_mapping)
        if max_s >= len(student_block_outputs) or max_t >= len(
                teacher_block_outputs):
            raise ValueError(
                'layer_mapping index out of range: '
                f'mapping={layer_mapping}, '
                f'student_blocks={len(student_block_outputs)}, '
                f'teacher_blocks={len(teacher_block_outputs)}')

        total_loss = torch.tensor(
            0.0, device=student_block_outputs[0].device)
        for s_idx, t_idx in layer_mapping:
            s_feat = student_block_outputs[s_idx]  # (B, T, D)
            t_feat = teacher_block_outputs[t_idx].detach()  # (B, T, D)
            # MSE per element, masked
            diff = (s_feat - t_feat) ** 2  # (B, T, D)
            diff = diff * mask  # mask out padding frames
            # Average over valid (frame, dim) entries
            loss = diff.sum() / (mask.sum() * s_feat.size(-1))
            total_loss = total_loss + loss
        total_loss = total_loss / len(layer_mapping)
        return total_loss

    def train(self, teacher_model, student_model, optimizer, data_loader,
              device, writer, args):
        """Train one epoch with feature alignment distillation.

        Args:
            teacher_model: frozen teacher (already .eval(), on device)
            student_model: trainable student (will be set to .train())
            optimizer: optimizer for student parameters
            data_loader: training DataLoader
            device: torch device
            writer: TensorboardX writer (may be None)
            args: dict with keys:
                - 'grad_clip', 'log_interval', 'epoch'
                - 'criterion' (str, e.g. 'ctc')
                - 'min_duration'
                - 'phase' (str, 'align' or 'finetune')
                - 'mse_weight' (float, weight for MSE in finetune phase)
                - 'layer_mapping' (list of tuples)
        """
        student_model.train()
        clip = args.get('grad_clip', 50.0)
        log_interval = args.get('log_interval', 10)
        epoch = args.get('epoch', 0)
        min_duration = args.get('min_duration', 0)
        phase = args.get('phase', 'align')
        mse_weight = args.get('mse_weight', 1.0)
        layer_mapping = args.get('layer_mapping', [(0, 1), (1, 2), (2, 3)])

        for batch_idx, batch_dict in enumerate(data_loader):
            feats = batch_dict['feats'].to(device)
            target = batch_dict['target'].to(device)
            target = target[:, 0] if target.shape[1] == 1 else target
            feats_lengths = batch_dict['feats_lengths'].to(device)
            label_lengths = batch_dict['target_lengths'].to(device)
            num_utts = feats_lengths.size(0)
            if num_utts == 0:
                continue

            # --- Teacher forward (frozen, no grad) ---
            with torch.no_grad():
                _, _, t_block_outputs = \
                    _unwrap_model(teacher_model) \
                    .forward_with_block_outputs(feats)

            # --- Student forward ---
            z_s, _, s_block_outputs = \
                _unwrap_model(student_model) \
                .forward_with_block_outputs(feats)

            # --- Block-level MSE loss ---
            mask = self._make_length_mask(feats_lengths, z_s.size(1))
            loss_mse = self.block_mse_loss(
                s_block_outputs, t_block_outputs, layer_mapping, mask)

            if phase == 'align':
                # Phase 1: pure feature alignment
                loss = loss_mse
                loss_ctc_val = torch.tensor(0.0, device=device)
                acc = 0.0
            else:
                # Phase 2: MSE + CTC
                loss_type = args.get('criterion', 'ctc')
                loss_ctc, acc = criterion(
                    loss_type, z_s, target, feats_lengths,
                    target_lengths=label_lengths,
                    min_duration=min_duration, validation=False)
                loss_ctc_val = loss_ctc
                loss = (mse_weight * loss_mse
                        + (1.0 - mse_weight) * loss_ctc)

            optimizer.zero_grad()
            loss.backward()
            grad_norm = clip_grad_norm_(student_model.parameters(), clip)
            if torch.isfinite(grad_norm):
                optimizer.step()

            if batch_idx % log_interval == 0:
                if phase == 'align':
                    logging.debug(
                        'TRAIN Epoch %d Batch %d  [align]  '
                        'mse %.6f  loss %.6f',
                        epoch, batch_idx, loss_mse.item(), loss.item())
                else:
                    logging.debug(
                        'TRAIN Epoch %d Batch %d  [finetune]  '
                        'mse %.6f  ctc %.6f  loss %.6f  acc %.6f  '
                        'mse_w %.2f',
                        epoch, batch_idx, loss_mse.item(),
                        loss_ctc_val.item(), loss.item(), acc, mse_weight)

    def cv(self, student_model, data_loader, device, args,
           teacher_model=None):
        """Cross-validation on the student model.

        Computes both MSE (feature alignment) and CTC losses.
        Returns CTC loss for LR scheduling.

        Returns:
            (cv_loss, cv_acc): CTC-based loss and accuracy.
        """
        student_model.eval()
        log_interval = args.get('log_interval', 10)
        epoch = args.get('epoch', 0)
        phase = args.get('phase', 'align')
        layer_mapping = args.get('layer_mapping', [(0, 1), (1, 2), (2, 3)])

        num_seen_utts = 1  # avoid /0
        total_ctc_loss = 0.0
        total_mse_loss = 0.0
        total_acc = 0.0

        with torch.no_grad():
            for batch_idx, batch_dict in enumerate(data_loader):
                feats = batch_dict['feats'].to(device)
                target = batch_dict['target'].to(device)
                target = target[:, 0] if target.shape[1] == 1 else target
                feats_lengths = batch_dict['feats_lengths'].to(device)
                label_lengths = batch_dict['target_lengths'].to(device)
                num_utts = feats_lengths.size(0)
                if num_utts == 0:
                    continue

                z_s, _, s_block_outputs = \
                    _unwrap_model(student_model) \
                    .forward_with_block_outputs(feats)

                # CTC loss (always compute for monitoring)
                loss_ctc, acc = criterion(
                    args.get('criterion', 'ctc'),
                    z_s, target, feats_lengths,
                    target_lengths=label_lengths,
                    min_duration=0, validation=True)

                # MSE loss
                mse_val = 0.0
                if teacher_model is not None:
                    _, _, t_block_outputs = \
                        _unwrap_model(teacher_model) \
                        .forward_with_block_outputs(feats)
                    mask = self._make_length_mask(feats_lengths,
                                                  z_s.size(1))
                    mse_val = self.block_mse_loss(
                        s_block_outputs, t_block_outputs,
                        layer_mapping, mask).item()

                if torch.isfinite(loss_ctc):
                    num_seen_utts += num_utts
                    total_ctc_loss += loss_ctc.item() * num_utts
                    total_mse_loss += mse_val * num_utts
                    total_acc += acc * num_utts

                if batch_idx % log_interval == 0:
                    logging.debug(
                        'CV Epoch %d Batch %d  ctc %.6f  mse %.6f  '
                        'acc %.6f  hist_ctc %.6f',
                        epoch, batch_idx, loss_ctc.item(), mse_val,
                        acc, total_ctc_loss / num_seen_utts)

        avg_ctc = total_ctc_loss / num_seen_utts
        avg_mse = total_mse_loss / num_seen_utts
        avg_acc = total_acc / num_seen_utts

        # In align phase, use MSE for scheduling; in finetune, use CTC
        if phase == 'align':
            cv_loss = avg_mse
        else:
            cv_loss = avg_ctc

        logging.info(
            'CV Epoch %d  cv_ctc %.6f  cv_mse %.6f  cv_acc %.6f  '
            '[scheduled on %s = %.6f]',
            epoch, avg_ctc, avg_mse, avg_acc,
            'mse' if phase == 'align' else 'ctc', cv_loss)
        return cv_loss, avg_acc
