# Copyright (c) 2021 Binbin Zhang
#               2026 Wayne (distillation extension)
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

"""Knowledge distillation executor for CTC-based KWS models.

This module provides DistillExecutor which trains a student model using
a combination of CTC loss (hard labels) and KL divergence loss (soft
labels from a frozen teacher model).

Loss formula:
    L = lambda * L_CTC(z_s, y) + (1-lambda) * T^2 * KL(p_t^T || p_s^T)

where:
    z_s, z_t: student / teacher frame-level logits (B, T, V)
    p^T = softmax(z / T)
    T: temperature
    lambda: weight balancing CTC and KD losses
"""

import logging

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from wekws.model.loss import criterion


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
            mask: (B, T, 1) float tensor, 1.0 for valid frames, 0.0 for padding.
        """
        # (1, T) < (B, 1)  =>  (B, T)
        mask = (torch.arange(max_len, device=lengths.device).unsqueeze(0)
                < lengths.unsqueeze(1))
        return mask.unsqueeze(-1).float()  # (B, T, 1)

    @staticmethod
    def kd_loss(z_s: torch.Tensor,
                z_t: torch.Tensor,
                mask: torch.Tensor,
                temperature: float) -> torch.Tensor:
        """Frame-level KL divergence between teacher and student.

        KL(p_t || p_s) where p = softmax(z / T), averaged over valid frames.

        Args:
            z_s: student logits (B, T, V)
            z_t: teacher logits (B, T, V)  -- detached
            mask: (B, T, 1) float mask
            temperature: softmax temperature

        Returns:
            Scalar KL loss (already divided by number of valid frames).
        """
        log_p_s = F.log_softmax(z_s / temperature, dim=-1)
        p_t = F.softmax(z_t / temperature, dim=-1)
        # kl_div expects input=log-prob, target=prob
        # reduction='none' => (B, T, V)
        kl = F.kl_div(log_p_s, p_t, reduction='none') * mask
        # average over all valid (frame, vocab) entries
        loss = kl.sum() / mask.sum()
        return loss

    def train(self, teacher_model, student_model, optimizer, data_loader,
              device, writer, args):
        """Train one epoch with knowledge distillation.

        Args:
            teacher_model: frozen teacher (already .eval(), on *device*)
            student_model: trainable student (will be set to .train())
            optimizer: optimizer for student parameters only
            data_loader: training DataLoader
            device: torch device
            writer: TensorboardX writer (may be None)
            args: dict with keys:
                - 'grad_clip', 'log_interval', 'epoch'
                - 'criterion' (str, e.g. 'ctc')
                - 'min_duration'
                - 'kd_temperature' (float)
                - 'kd_lambda' (float, current epoch's lambda)
        """
        student_model.train()
        clip = args.get('grad_clip', 50.0)
        log_interval = args.get('log_interval', 10)
        epoch = args.get('epoch', 0)
        min_duration = args.get('min_duration', 0)
        temperature = args.get('kd_temperature', 2.0)
        lam = args.get('kd_lambda', 0.5)

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
                z_t, _ = teacher_model(feats)

            # --- Student forward ---
            z_s, _ = student_model(feats)

            # --- CTC loss (on student logits) ---
            loss_type = args.get('criterion', 'ctc')
            loss_ctc, acc = criterion(loss_type,
                                      z_s,
                                      target,
                                      feats_lengths,
                                      target_lengths=label_lengths,
                                      min_duration=min_duration,
                                      validation=False)

            # --- KD loss ---
            if lam < 1.0:
                mask = self._make_length_mask(feats_lengths,
                                              z_s.size(1))
                loss_kd = self.kd_loss(z_s, z_t, mask, temperature)
                loss = (lam * loss_ctc
                        + (1.0 - lam) * temperature * temperature * loss_kd)
            else:
                loss_kd = torch.tensor(0.0, device=device)
                loss = loss_ctc

            optimizer.zero_grad()
            loss.backward()
            grad_norm = clip_grad_norm_(student_model.parameters(), clip)
            if torch.isfinite(grad_norm):
                optimizer.step()

            if batch_idx % log_interval == 0:
                logging.debug(
                    'TRAIN Epoch %d Batch %d  loss %.6f  '
                    'ctc %.6f  kd %.6f  acc %.6f  lambda %.2f  T %.1f',
                    epoch, batch_idx, loss.item(),
                    loss_ctc.item(), loss_kd.item(),
                    acc, lam, temperature)

    def cv(self, student_model, data_loader, device, args,
           teacher_model=None):
        """Cross-validation on the student model.

        When *teacher_model* is provided the KD loss is also computed and
        logged, but only the CTC loss is used for LR scheduling / early
        stopping (consistent with the original executor).

        Returns:
            (cv_loss, cv_acc): CTC-based loss and accuracy.
        """
        student_model.eval()
        log_interval = args.get('log_interval', 10)
        epoch = args.get('epoch', 0)
        temperature = args.get('kd_temperature', 2.0)

        num_seen_utts = 1  # avoid /0
        total_loss = 0.0
        total_acc = 0.0
        total_kd = 0.0

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

                z_s, _ = student_model(feats)
                loss_ctc, acc = criterion(
                    args.get('criterion', 'ctc'),
                    z_s, target, feats_lengths,
                    target_lengths=label_lengths,
                    min_duration=0,
                    validation=True)

                kd_val = 0.0
                if teacher_model is not None:
                    z_t, _ = teacher_model(feats)
                    mask = self._make_length_mask(feats_lengths,
                                                  z_s.size(1))
                    kd_val = self.kd_loss(z_s, z_t, mask,
                                          temperature).item()

                if torch.isfinite(loss_ctc):
                    num_seen_utts += num_utts
                    total_loss += loss_ctc.item() * num_utts
                    total_acc += acc * num_utts
                    total_kd += kd_val * num_utts

                if batch_idx % log_interval == 0:
                    logging.debug(
                        'CV Epoch %d Batch %d  ctc %.6f  kd %.6f  '
                        'acc %.6f  hist_loss %.6f',
                        epoch, batch_idx, loss_ctc.item(), kd_val,
                        acc, total_loss / num_seen_utts)

        avg_loss = total_loss / num_seen_utts
        avg_acc = total_acc / num_seen_utts
        avg_kd = total_kd / num_seen_utts
        logging.info(
            'CV Epoch %d  cv_ctc_loss %.6f  cv_kd_loss %.6f  cv_acc %.6f',
            epoch, avg_loss, avg_kd, avg_acc)
        return avg_loss, avg_acc
