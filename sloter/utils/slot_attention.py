from torch import nn
import torch
import math
from PIL import Image
import numpy as np
import torch.nn.functional as F


class SlotAttention(nn.Module):
    def __init__(self, num_classes, slots_per_class, dim, iters=3, eps=1e-8, vis=False, vis_id=0, loss_status=1, power=1, to_k_layer=1):
        super().__init__()
        self.num_classes = num_classes
        self.slots_per_class = slots_per_class
        self.num_slots = num_classes * slots_per_class
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5
        self.loss_status = loss_status

        # 초기 slot을 평균과 표준편차를 이용해 정규분포에서 샘플링하여 초기화
        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_sigma = nn.Parameter(torch.randn(1, 1, dim))

        self.to_q = nn.Sequential(nn.Linear(dim, dim))
        to_k_layer_list = [nn.Linear(dim, dim)]
        for _ in range(1, to_k_layer):
            to_k_layer_list.append(nn.ReLU(inplace=True))
            to_k_layer_list.append(nn.Linear(dim, dim))
        self.to_k = nn.Sequential(*to_k_layer_list)
        self.gru = nn.GRU(dim, dim)

        self.vis = vis
        self.vis_id = vis_id
        self.power = power

    def forward(self, inputs, inputs_x):
        b, n, d = inputs.shape

        mu = self.slots_mu.expand(b, self.num_slots, -1)
        sigma = self.slots_sigma.expand(b, self.num_slots, -1)
        slots = torch.normal(mu, sigma.abs())

        k, v = self.to_k(inputs), inputs
        updates = None

        for _ in range(self.iters):
            slots_prev = slots
            q = self.to_q(slots)
            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale

            # Softmax attention
            attn = torch.softmax(dots, dim=-1)
            updates = torch.einsum('bjd,bij->bid', inputs_x, attn)
            updates = updates / inputs_x.size(2)

            self.gru.flatten_parameters()
            slots, _ = self.gru(updates.reshape(1, -1, d), slots_prev.reshape(1, -1, d))
            slots = slots.reshape(b, -1, d)

            if self.vis:
                slots_vis = attn.clone()
                if self.slots_per_class > 1:
                    new_slots_vis = torch.zeros((slots_vis.size(0), self.num_classes, slots_vis.size(-1))).to(slots_vis.device)
                    for slot_class in range(self.num_classes):
                        new_slots_vis[:, slot_class] = torch.sum(
                            slots_vis[:, self.slots_per_class * slot_class: self.slots_per_class * (slot_class + 1)],
                            dim=1
                        )
                    slots_vis = new_slots_vis

        if updates is None:
            raise RuntimeError("SlotAttention forward failed: no update produced. Likely iters=0.")

        if self.slots_per_class > 1:
            new_updates = torch.zeros((updates.size(0), self.num_classes, updates.size(-1))).to(updates.device)
            for slot_class in range(self.num_classes):
                new_updates[:, slot_class] = torch.sum(
                    updates[:, self.slots_per_class * slot_class: self.slots_per_class * (slot_class + 1)],
                    dim=1
                )
            updates = new_updates

        attn_relu = torch.relu(attn)
        slot_loss = torch.sum(attn_relu) / (attn.size(0) * attn.size(1) * attn.size(2))

        return self.loss_status * torch.sum(updates, dim=2), torch.pow(slot_loss, self.power)
