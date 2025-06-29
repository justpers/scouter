import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import create_model
from collections import OrderedDict

from sloter.utils.slot_attention import SlotAttention
from sloter.utils.position_encode import build_position_encoding


class Identical(nn.Module):
    """A dummy layer that returns its input unchanged (used to strip heads)."""

    def forward(self, x):  # pylint: disable=arguments-differ
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Backbone loader
# ──────────────────────────────────────────────────────────────────────────────

def load_backbone(args):
    """Create a *timm* backbone and optionally load no‑slot checkpoint weights.

    The function also strips the classification head when **use_slot** is *True*
    **and** the user does *not* wish to fine‑tune the backbone (``args.grad`` is
    *False*).
    """

    backbone = create_model(
        args.model,
        pretrained=args.pre_trained,
        num_classes=args.num_classes,
    )

    # ── MNIST 흑백 지원 ────────────────────────────────────────────────────────
    if args.dataset == "MNIST":
        backbone.conv1 = nn.Conv2d(1, 64, 3, stride=2, padding=1, bias=False)

    # ── no‑slot 학습된 파라미터 불러오기 (선택) ─────────────────────────────────
    if args.use_slot and args.use_pre:
        ckpt_path = f"saved_model/{args.dataset}_no_slot_checkpoint.pth"
        ckpt = torch.load(ckpt_path, map_location="cpu")
        # remove "backbone." prefix added during *SlotModel* training
        new_state = OrderedDict((k[9:], v) for k, v in ckpt["model"].items())
        backbone.load_state_dict(new_state, strict=False)
        print("✔ no‑slot backbone weights loaded")

    # ── Strip the classification head if we are going to attach SlotAttention ─
    if args.use_slot and not args.grad:
        if "seresnet" in args.model:
            backbone.avg_pool, backbone.last_linear = Identical(), Identical()
        elif "res" in args.model:
            backbone.global_pool, backbone.fc = Identical(), Identical()
        elif "efficient" in args.model:
            backbone.global_pool, backbone.classifier = Identical(), Identical()
        elif "densenet" in args.model:
            backbone.global_pool, backbone.classifier = Identical(), Identical()
        elif "mobilenet" in args.model:
            backbone.global_pool = Identical()
            backbone.conv_head = Identical()
            backbone.act2 = Identical()
            backbone.classifier = Identical()

    return backbone


# ──────────────────────────────────────────────────────────────────────────────
# SlotModel
# ──────────────────────────────────────────────────────────────────────────────

class SlotModel(nn.Module):
    """ResNet/ResNeSt/EfficientNet + (optional) SlotAttention classifier.

    The model is compatible with *reproduce_SCOUTER* evaluation code: the
    ``forward`` signature accepts **softmax**, **save_id**, and **sens**
    parameters, while preserving the training logic of the original SCOUTER
    implementation.
    """

    def __init__(self, args):
        super().__init__()
        self.use_slot = args.use_slot

        # ── Backbone ─────────────────────────────────────────────────────────
        self.backbone = load_backbone(args)

        # *timm* models expose the dim through `num_features`. Fallback to the
        # last Conv‑layer's *in_channels* for exotic architectures.
        self.backbone_out_dim = getattr(self.backbone, "num_features", None)
        if self.backbone_out_dim is None:
            # last child that is *not* an nn.Identity
            for m in reversed(list(self.backbone.modules())):
                if isinstance(m, nn.Conv2d):
                    self.backbone_out_dim = m.in_channels
                    break
            if self.backbone_out_dim is None:
                raise RuntimeError("Could not infer backbone output channels.")

        # record for external access (some eval scripts query these attr.)
        self.channel = self.backbone_out_dim
        self.feature_size = None  # will be set on the first forward pass

        # ── SlotAttention branch ─────────────────────────────────────────────
        if self.use_slot:
            self.conv1x1 = nn.Conv2d(self.backbone_out_dim, args.hidden_dim, 1, bias=False)
            self.position_emb = build_position_encoding("sine", hidden_dim=args.hidden_dim)
            self.lambda_value = float(args.lambda_value)

            self.slot = SlotAttention(
                num_classes=args.num_classes,
                slots_per_class=args.slots_per_class,
                dim=args.hidden_dim,
                vis=args.vis,
                vis_id=args.vis_id,
                loss_status=args.loss_status,
                power=args.power,
                to_k_layer=args.to_k_layer,
            )

        # ── (Optional) layer freezing ───────────────────────────────────────
        if args.pre_trained:
            self._freeze_layers(self.backbone, args.freeze_layers)

    # ---------------------------------------------------------------------
    #  Utility
    # ---------------------------------------------------------------------

    @staticmethod
    def _freeze_layers(module: nn.Module, freeze_layer_num: int):
        """Freeze *layer1* · *layer2* · *layer3* · *layer4* bottom‑up."""
        if freeze_layer_num == 0:
            return
        unfreeze = ["layer4", "layer3", "layer2", "layer1"][: 4 - freeze_layer_num]
        for name, child in module.named_children():
            if any(layer in name for layer in unfreeze):
                continue
            for p in child.parameters():
                p.requires_grad = False
            SlotModel._freeze_layers(child, freeze_layer_num)

    # ---------------------------------------------------------------------
    #  Forward
    # ---------------------------------------------------------------------

    def _extract_feat(self, x):
        """Return a 4‑D feature map (B,C,H,W) from the backbone."""
        if hasattr(self.backbone, "forward_features"):
            return self.backbone.forward_features(x)
        # Fallback: some backbones return logits; we hook before GAP when no
        # SlotAttention is used. For SlotAttention, we *require* spatial map.
        return self.backbone(x)

    def forward(
        self,
        x,
        target=None,
        *,
        softmax: bool = False,
        save_id=None,
        sens: int = -1,
    ):
        """Forward pass compatible with *reproduce_SCOUTER* evaluation code.

        Parameters
        ----------
        x : torch.Tensor
            Input image tensor of shape (B,3,H,W).
        target : torch.LongTensor | None
            Ground‑truth labels; if *None*, the model is in inference mode.
        softmax : bool, default = False
            If *True*, return *probabilities* (``softmax``) instead of log‑probs.
        save_id : Any, optional
            Identifier forwarded to *SlotAttention* for visualization.
        sens : int, default = -1
            Sensitivity‑analysis flag (kept for API compatibility).
        """

        feat = self._extract_feat(x)  # (B,C,H,W)
        if feat.dim() != 4:
            raise RuntimeError(
                "Backbone did not return a 4‑D feature map – check model choice "
                "or override 'forward_features'."
            )

        if self.feature_size is None:
            # Cache H=W on first run (assumes square feature map)
            self.feature_size = feat.shape[-1]

        # ── SlotAttention branch ────────────────────────────────────────────
        if self.use_slot:
            feat = torch.relu(self.conv1x1(feat))  # (B,D,H,W)
            feat_pe = feat + self.position_emb(feat)

            B, D, H, W = feat.shape
            tokens = feat.reshape(B, D, -1).permute(0, 2, 1)  # (B,H*W,D)
            tokens_pe = feat_pe.reshape(B, D, -1).permute(0, 2, 1)

            # *SlotAttention* API changed between versions. Try the new API
            # first (with save_id & sens); if that fails, fall back.
            try:
                logits, attn_loss = self.slot(tokens_pe, tokens, save_id, sens)
            except TypeError:
                logits, attn_loss = self.slot(tokens_pe, tokens)
        # ── No‑slot baseline ────────────────────────────────────────────────
        else:
            # Global Average Pooling
            logits = feat.mean(dim=(2, 3))  # (B,C)

        # ────────────────────────────────────────────────────────────────────
        # Output post‑processing                                                 
        # ────────────────────────────────────────────────────────────────────
        if softmax:
            output = F.softmax(logits, dim=1)
        else:
            output = F.log_softmax(logits, dim=1)

        # ── Training mode: return loss tuple(s) ─────────────────────────────
        if target is not None:
            ce_loss = F.nll_loss(output, target) if not softmax else F.cross_entropy(logits, target)
            if self.use_slot:
                total_loss = ce_loss + self.lambda_value * attn_loss
                return output, [total_loss, ce_loss, attn_loss]
            else:
                return output, [ce_loss, ce_loss]

        # ── Inference mode ─────────────────────────────────────────────────
        return output