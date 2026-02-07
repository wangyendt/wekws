# 从 top20 Head 到更小 Backbone：蒸馏路线与配置建议

下面内容是把我上一条回复整理成一个可直接保存/版本管理的 Markdown 文档，避免你在聊天窗口复制时公式被破坏。

---

## 1. 现状判断：Head 已经不是主要压缩对象

你已经把 Head 从 2599 裁到 top20（含唤醒词），参数从约 752K 降到约 390K，且性能与 baseline 基本一致。下一步要再变小，主要只能动 **Backbone**。

---

## 2. 老师用 base.pt 还是 top20.pt？

推荐：**老师用 top20.pt**。

原因：

1. **输出空间一致**：老师/学生都在 20 类空间上输出，蒸馏的 KL/logit matching 是同一件事。  
2. top20.pt 已经验证“性能与 baseline 一致”，它就是最终任务的强老师。  
3. base.pt 的 2599 维输出携带大量与最终部署无关的分布；如果学生最终只保留 20 维，直接蒸馏 base 的 full-softmax 对齐成本更高、坑更多。

---

## 3. 学生用 20-head 还是先 2599-head 再手术？

默认推荐：**学生从第一天就用 20-head（与老师一致）**。

也就是说：不要走“先 2599-head → 蒸馏 → 再权重手术”这条复杂路线。理由：

- 你最终部署就是 20 输出空间，让学生从一开始就学最终目标最干净。  
- 少一次“中途切 head/换 dict/改评测链路”，就少一类致命坑。  
- 你此前已经验证 20-head 足够表达任务。

### 什么时候才考虑“两段式（先 2599-head 再手术）”？

当你把学生 backbone 缩得非常狠，出现明显掉点，且你怀疑“学生需要先学更通用的声学结构”时，可以考虑：

- 阶段 A（可选）：学生先用 2599-head，用 base.pt 做更通用的蒸馏（更像预训练）。  
- 阶段 B：对学生做一次权重手术裁到 20，然后用 top20.pt 做最终任务蒸馏 + CTC 微调。

这条路更复杂，默认不要一上来就走。

---

## 4. CTC 蒸馏怎么做：CTC + Logit/KL 蒸馏（推荐）

你当前是 CTC 训练。最稳的 KD loss 形式是：

$$
\mathcal L=\lambda\,\mathcal L_{\text{CTC}}(z_s,y)+(1-\lambda)\,T^2\,\mathrm{KL}(p_t^T\Vert p_s^T)
$$

其中：

- $z_s,z_t$：学生/老师每帧 logits，形状 $B\times T\times V$，这里 $V=20$  
- $p^T=\mathrm{softmax}(z/T)$  
- $T$：温度，常用 2 到 4  
- $\lambda$：CTC 与 KD 权重，建议从 0.7 起步，后期降到 0.5  

关键细节：

- KL 要按有效帧做 mask（语音长度不同）。  
- 蒸馏时包含 blank 类一起蒸（CTC 的关键）。

---

## 5. 训练配方：按“最稳成功率”的顺序走

### Step 0：学生的特征管线先别乱改

为了让老师输出与学生输入一致，建议学生先沿用你 FSMN 的特征管线（80 fbank + context expansion + frame_skip=3 等）。  
不要一上来就切换到另一套配置（例如 mel 维度、loss、frame_skip 都不同），变量太多会让你难定位问题。

### Step 1：先做一个 FSMN-mini 学生（最稳）

先在现有 FSMN 结构上做递减裁剪，比直接换到 mdtc/tcn/gru 稳得多。

建议每次只动一点，按这个递进试：

- num_layers：4 → 3  
- linear_dim：250 → 192 → 160  
- proj_dim：128 → 96 → 64  
- input_affine_dim/output_affine_dim：140 → 112 → 96  

一般参数量会近似按维度平方缩，能很快下一个台阶。

### Step 2：蒸馏训练（主训练阶段）

- 老师：top20.pt（冻结，eval 模式）  
- 学生：20-head + 小 backbone  
- loss：CTC + KL（上面的公式）  
- $\lambda$：前 20 epoch 用 0.7，后面用 0.5  
- $T$：2 或 4（建议先 2，不行再 4）

### Step 3：去掉 KD，纯 CTC 收尾 5 到 10 epoch

最后只用 CTC 收尾微调，让学生更贴近真实标签分布，通常 FAR 会更稳。

---

## 6. “蒸馏后再权重手术”到底何时需要？

- 如果学生从一开始就是 20-head：不需要权重手术。  
- 只有当你走“两段式（先 2599-head）”：才需要在阶段 A→B 中间做一次权重手术把学生 head 裁到 20。

---

## 7. 最小改动的蒸馏伪代码（训练循环里加这段就够用）

```python
# teacher: frozen, eval()
with torch.no_grad():
    z_t, _ = teacher(feats, feats_lens)   # (B,T,V=20)

z_s, _ = student(feats, feats_lens)       # (B,T,V=20)

loss_ctc = ctc_loss(z_s, target, feats_lens, target_lens)

T = 2.0
log_p_s = torch.log_softmax(z_s / T, dim=-1)
p_t = torch.softmax(z_t / T, dim=-1)

# mask: (B,T,1)
mask = make_len_mask(feats_lens, max_len=z_s.size(1)).unsqueeze(-1).to(z_s.device)

loss_kd = (torch.nn.functional.kl_div(log_p_s, p_t, reduction="none") * mask).sum() / mask.sum()
loss = lam * loss_ctc + (1 - lam) * (T * T) * loss_kd
```

---

## 8. 最该做的破局动作（简版）

1. 默认让学生从一开始就用 **20-head**，老师用 **top20.pt**。  
2. 先做 **FSMN-mini** 结构递减，不要急着换到 mdtc/tcn/gru。  
3. 用 **CTC + KL 蒸馏** 主训，再用 **纯 CTC** 收尾。

---

如果你告诉我你的压缩目标（例如参数 < 200K、延迟 < X ms、FAR@1h < Y），我可以把 FSMN 的递减档位做成一张配置表：每档改哪些维度、预计缩多少、风险点和训练超参建议。
