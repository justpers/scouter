# sloter/utils/slot_attention.py
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SlotAttention(nn.Module):
    """
    간소화-버전 Slot-Attention
      • dots → softmax 정규화 (안정적)
      • iters ≥ 3 권장
      • 초기 slot 가중치 Xavier-uniform
    """
    def __init__(
        self,
        num_classes: int,
        slots_per_class: int,
        dim: int,
        iters: int = 3,
        vis: bool = False,
        vis_id: int = 0,
        loss_status: float = 1.0,
        power: float = 1.0,
        to_k_layer: int = 1,
    ) -> None:
        super().__init__()

        # ─────────── 기본 파라미터 ───────────
        self.num_classes     = num_classes
        self.slots_per_class = slots_per_class
        self.num_slots       = num_classes * slots_per_class
        self.iters           = max(1, iters)
        self.scale           = dim ** -0.5
        self.loss_status     = loss_status
        self.power           = power
        self.eps             = 1e-8                       # ★ 추가 ✔
        self.cls_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes, bias=False)
        )

        # ─────────── 초기 slots ───────────
        self.initial_slots = nn.Parameter(torch.empty(1, self.num_slots, dim))
        nn.init.xavier_uniform_(self.initial_slots)

        # ─────────── 프로젝션 레이어 ───────────
        self.to_q = nn.Linear(dim, dim)

        to_k_layers = [nn.Linear(dim, dim)]
        for _ in range(1, to_k_layer):
            to_k_layers += [nn.ReLU(inplace=True), nn.Linear(dim, dim)]
        self.to_k = nn.Sequential(*to_k_layers)

        # GRU (batch_first=False → [seq, batch, feat])
        self.gru = nn.GRU(dim, dim, batch_first=False)

        # (옵션) 시각화
        self.vis    = vis
        self.vis_id = vis_id

    # -----------------------------------------------------
    def forward(
        self, pos_flat: torch.Tensor, feats_flat: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        pos_flat   : [B, HW, D] positional embeddings
        feats_flat : [B, HW, D] visual features

        Returns
        -------
        logits     : [B, num_classes]
        attn_loss  : scalar
        """
        B, N, D = feats_flat.shape

        k = self.to_k(pos_flat)            # [B, N, D]
        v = feats_flat                     # [B, N, D]

        # ---------- 초기 slot 설정 ----------
        slots = self.initial_slots.expand(B, -1, -1).contiguous()

        # ---------- 반복 업데이트 ----------
        for _ in range(self.iters):
            q = self.to_q(slots)                              # [B, S, D]
            dots = torch.einsum("bsd,bnd->bsn", q, k) * self.scale
            attn = torch.softmax(dots, dim=2) + self.eps      # 안정화용 eps

            updates = torch.einsum("bnd,bsn->bsd", v, attn)   # [B, S, D]
            updates = updates / N                             # mean-pool

            # GRU expects [seq, batch, feat]
            self.gru.flatten_parameters()
            slots, _ = self.gru(
                updates.unsqueeze(0),                         # [1, B*S, D]
                slots.reshape(1, B * self.num_slots, D),      # hidden
            )
            slots = slots.squeeze(0).view(B, self.num_slots, D)

        # ---------- 클래스 단위로 합치기 ----------
        if self.slots_per_class > 1:
            slots = slots.view(B, self.num_classes, self.slots_per_class, D).sum(2)

        # ---------- 로짓 & 규제 항 ----------
        #logits = slots.mean(2)                                # [B, num_classes]

        slot_repr = slots.mean(2)                    # [B, num_classes, D]
        logits    = self.cls_head(slot_repr)         # [B, num_classes]

        attn_loss = (-attn * (attn + self.eps).log()).mean()  # entropy loss
        attn_loss = torch.pow(attn_loss, self.power) * self.loss_status
        return logits, attn_loss
