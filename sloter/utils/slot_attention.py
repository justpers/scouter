import os
from typing import Optional, Sequence

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F


class SlotAttention(nn.Module):
    """Slot‑Attention module customised for SCOUTER & reproduce_SCOUTER.

    *   Keeps the original behaviour used during training/visualisation.
    *   Adds the extra arguments `save_id` and `sens` required by the public
        evaluation code, so you can drop‑in replace the file without touching
        the rest of the pipeline.
    *   Works for both **single‑slot** (``slots_per_class == 1``) and
        **multi‑slot** settings.
    *   If ``self.vis`` **or** ``save_id`` **or** ``sens > -1`` is active,
        intermediate attention maps are normalised to ``0‑255`` and saved as
        8‑bit PNGs (grayscale) just like the original implementation – but the
        output path follows the logic expected by *reproduce_SCOUTER*.
    """

    def __init__(
        self,
        num_classes: int,
        slots_per_class: int,
        dim: int,
        *,
        iters: int = 3,
        eps: float = 1e-8,
        vis: bool = False,
        vis_id: int = 0,
        loss_status: float = 1.0,
        power: float = 1.0,
        to_k_layer: int = 1,
    ) -> None:
        super().__init__()

        # ─────────────────── hyper‑parameters ────────────────────
        self.num_classes = num_classes
        self.slots_per_class = slots_per_class
        self.num_slots = num_classes * slots_per_class
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5
        self.loss_status = loss_status
        self.power = power

        # ─────────────────── slot initialisation ─────────────────
        slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        slots_sigma = nn.Parameter(torch.randn(1, 1, dim))
        # Broadcast so every slot gets its own mean/σ sample
        mu = slots_mu.expand(1, self.num_slots, -1)
        sigma = slots_sigma.expand(1, self.num_slots, -1).abs()
        self.initial_slots = nn.Parameter(torch.normal(mu, sigma))

        # ─────────────────── projections & GRU ───────────────────
        self.to_q = nn.Linear(dim, dim)  # unused but kept for BC

        to_k: list[nn.Module] = [nn.Linear(dim, dim)]
        for _ in range(1, to_k_layer):
            to_k += [nn.ReLU(inplace=True), nn.Linear(dim, dim)]
        self.to_k = nn.Sequential(*to_k)

        self.gru = nn.GRU(dim, dim)

        # ─────────────────── vis & misc flags ────────────────────
        self.vis = vis
        self.vis_id = vis_id

    # -------------------------------------------------------------
    # forward
    # -------------------------------------------------------------
    def forward(
        self,
        inputs: torch.Tensor,  # (B, N, D) positional‑encoded tokens
        inputs_x: torch.Tensor,  # (B, N, D) raw tokens for value aggregation
        save_id: Optional[Sequence[int]] = None,
        sens: int = -1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run slot‑attention.

        Args:
            inputs:   PE‑added tokens (B, N, D)
            inputs_x: raw tokens (B, N, D) – used for weighted aggregation
            save_id:  ``(gt_class, worst_class, save_root, file_stem)`` used by
                       reproduce_SCOUTER for qualitative export. Pass *None* to
                       disable.
            sens:     If >=0, save the attention map of this slot index as
                       ``noisy.png`` (used by the sensitivity metric).
        Returns:
            (logits, slot_loss) tuple exactly as expected by the upstream code.
        """
        B, N, D = inputs.shape
        # (B, S, D) – each batch gets its own copy of the learnable template.
        slots = self.initial_slots.expand(B, -1, -1)

        # Pre‑compute keys once (dot‑product attention w.r.t. fixed queries)
        k = self.to_k(inputs)  # (B, N, D)

        # NOTE: we keep *v = inputs* semantics for clarity but use inputs_x for
        #       the actual value aggregation (matches original implementation).
        v = inputs

        for _ in range(self.iters):
            slots_prev = slots

            # Original paper projects slots to queries. The first SCOUTER impl.
            # skipped the projection; we keep that choice for reproducibility.
            q = slots  # or self.to_q(slots)

            # Attention logits & normalisation (scaled dot‑product variant)
            dots = torch.einsum("bid,bjd->bij", q, k) * self.scale  # (B,S,N)
            # Row‑wise & batch‑wise normalisation trick from SCOUTER codebase
            norm = dots.sum(2, keepdim=True)  # (B,S,1)
            dots = (dots / norm) * norm.sum(1, keepdim=True)  # keeps magnitude
            attn = torch.sigmoid(dots)  # (B,S,N)

            # Aggregate inputs_x with the attention weights
            updates = torch.einsum("bjd,bij->bid", inputs_x, attn) / inputs_x.size(1)

            # GRU update (slot_dim acts as hidden_size)
            self.gru.flatten_parameters()
            slots, _ = self.gru(
                updates.reshape(1, -1, D),  # (1, B*S, D)
                slots_prev.reshape(1, -1, D),
            )
            slots = slots.reshape(B, -1, D)

            # Save attn for visualisation outside the loop
            if self.vis or save_id or sens > -1:
                slots_vis = attn.clone()  # (B,S,N)

        # ---------------------------------------------------------
        # Optional visualisation / export step
        # ---------------------------------------------------------
        if self.vis or save_id or sens > -1:
            # Collapse multiple slots of the same class if applicable
            if self.slots_per_class > 1:
                new_vis = torch.zeros((slots_vis.size(0), self.num_classes, slots_vis.size(-1)), device=slots_vis.device)
                for c in range(self.num_classes):
                    start = c * self.slots_per_class
                    end = (c + 1) * self.slots_per_class
                    new_vis[:, c] = slots_vis[:, start:end].sum(1)
                slots_vis = new_vis  # (B, C, N)

            vis_single = slots_vis[self.vis_id]  # (C, N)
            vis_single = vis_single.detach()
            vis_single = (vis_single - vis_single.min()) / (vis_single.max() - vis_single.min() + 1e-12)
            vis_single = (vis_single * 255.0).to(torch.uint8)
            side = int(vis_single.size(1) ** 0.5)
            vis_single = vis_single.view(vis_single.size(0), side, side).cpu().numpy()

            for idx, img in enumerate(vis_single):
                pil = Image.fromarray(img, mode="L")
                # --- reproduce_SCOUTER export rules -------------------
                if save_id is not None:
                    gt_cls, worst_cls, root_dir, fname = save_id
                    if idx == gt_cls:
                        pil.save(os.path.join(root_dir, "positive", f"{fname}.png"))
                    elif idx == worst_cls:
                        pil.save(os.path.join(root_dir, "negative", f"{fname}.png"))
                elif sens > -1 and idx == sens:
                    pil.save("noisy.png")
                elif self.vis:
                    # Fallback visualisation path
                    os.makedirs("sloter/vis", exist_ok=True)
                    pil.save(f"sloter/vis/slot_{idx}.png")

        # ---------------------------------------------------------
        # Loss & logits aggregation
        # ---------------------------------------------------------
        if self.slots_per_class > 1:
            merged = torch.zeros((updates.size(0), self.num_classes, updates.size(-1)), device=updates.device)
            for c in range(self.num_classes):
                start = c * self.slots_per_class
                end = (c + 1) * self.slots_per_class
                merged[:, c] = updates[:, start:end].sum(1)
            updates = merged  # (B, C, D)

        # slot‑level sparsity regularisation (same as original)
        attn_relu = F.relu(attn)
        slot_loss = attn_relu.sum() / (attn.size(0) * attn.size(1) * attn.size(2))

        logits = self.loss_status * updates.sum(2)  # (B, C)
        return logits, torch.pow(slot_loss, self.power)