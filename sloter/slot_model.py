import torch
import torch.nn as nn
import torch.nn.functional as F
from sloter.utils.slot_attention import SlotAttention
from sloter.utils.position_encode import build_position_encoding
from timm.models import create_model
from collections import OrderedDict


class Identical(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x


def load_backbone(args):
    # 1) 분류 헤드 포함해 뽑고, 나중에 필요하면 지워줄 거임
    bone = create_model(
        args.model,
        pretrained=args.pre_trained,
        num_classes=args.num_classes,
        in_chans=getattr(args, "input_channels", 3)
    )

    # 2) grayscale → 첫 conv 교체
    if getattr(args, "input_channels", 3) == 1 and bone.conv1.in_channels == 3:
        bone.conv1 = nn.Conv2d(1, bone.conv1.out_channels,
                               kernel_size=bone.conv1.kernel_size,
                               stride=bone.conv1.stride,
                               padding=bone.conv1.padding,
                               bias=False)

    # 3) 만약 use_slot 이면 언제나 classifier 헤드를 전부 Identity 로 교체
    if args.use_slot:
        for name in ["global_pool", "fc", "classifier", "last_linear", "head", "head_fc"]:
            if hasattr(bone, name):
                setattr(bone, name, Identical())

    # 4) pretrained backbone 불러오기 옵션
    if args.use_pre:
        ckpt = torch.load(f"saved_model/{args.dataset}_no_slot_checkpoint.pth")
        new_st = OrderedDict()
        for k, v in ckpt["model"].items():
            nm = k.replace("backbone.", "")
            if nm in bone.state_dict() and bone.state_dict()[nm].shape == v.shape:
                new_st[nm] = v
        bone.load_state_dict(new_st, strict=False)
        print("▶ Loaded pretrained backbone parameters")

        if not args.grad and not args.use_slot:
            # grad freeze only when slot 안 쓸 때
            # (slot 쓸 때는 feature map 필요하므로 conv1~layer4 freeze만)
            bone.global_pool = Identical()
            bone.fc = Identical()

    return bone

class SlotModel(nn.Module):
    # --------------------------- INIT ----------------------------------
    def __init__(self, args):
        super().__init__()
        self.use_slot = args.use_slot
        self.hidden_dim = args.hidden_dim
        self.lambda_value = float(args.lambda_value)

        # 1) backbone ----------------------------------------------------
        self.backbone = load_backbone(args)

        # 2) slot branch -------------------------------------------------
        if self.use_slot:
            # (a) conv1x1는 백본 출력 채널을 본 뒤에 만들기 위해 placeholder
            self.conv1x1 = None   # -> real module은 첫 forward 때 생성

            # (b) positional encoding
            self.position_emb = build_position_encoding(
                'sine', hidden_dim=self.hidden_dim
            )

            # (c) Slot-Attention
            self.slot = SlotAttention(
                num_classes      = args.num_classes,
                slots_per_class  = args.slots_per_class,
                dim              = self.hidden_dim,
                iters            = getattr(args, "slot_iters", 3),   # ★ 반드시 1 이상
                vis              = args.vis,
                vis_id           = args.vis_id,
                loss_status      = args.loss_status,
                power            = args.power,
                to_k_layer       = args.to_k_layer
            )

        # 3) freeze 일부 레이어(선택) ------------------------------------
        if args.pre_trained:
            self._freeze_backbone(args.freeze_layers)

    # ------------------------- UTILITIES -------------------------------
    def _freeze_backbone(self, freeze_layer_num):
        stages = ['layer1', 'layer2', 'layer3', 'layer4'][:freeze_layer_num]

        def dfs(m):
            for n, c in m.named_children():
                if n in stages:
                    for p in c.parameters():
                        p.requires_grad = False
                else:
                    dfs(c)
        dfs(self.backbone)

    def _ensure_conv1x1(self, feats):
        "몇 번째 forward 가 되었든 conv1x1 이 없으면 지금 만들어 삽입"
        if self.conv1x1 is None:
            in_ch = feats.size(1)            # runtime 에서 채널 확보
            self.conv1x1 = nn.Conv2d(in_ch, self.hidden_dim, kernel_size=1).to(feats.device)

    # --------------------------- FORWARD -------------------------------
    @torch.no_grad()
    def _backbone_features(self, x):
        # 백본이 timm / torchvision 등에 따라 forward_features 유무가 다름
        if self.use_slot and hasattr(self.backbone, "forward_features"):
            return self.backbone.forward_features(x)      # [B,C,H,W]
        return self.backbone(x)                           # [B,num_cls] or [B,C,H,W]

    def forward(self, x, target=None):
        feats = self._backbone_features(x)

        # ----------------- SLOT MODE -----------------------------------
        if self.use_slot:
            self._ensure_conv1x1(feats)                   # <- 동적 생성/검증
            feats = F.relu(self.conv1x1(feats), inplace=True)

            pos   = self.position_emb(feats)              # [B,C,H,W]

            B, C, H, W = feats.shape
            feats_flat = feats.flatten(2).permute(0,2,1)  # [B,HW,C]
            pos_flat   = pos  .flatten(2).permute(0,2,1)

            logits, attn_loss = self.slot(pos_flat, feats_flat)
            output = F.log_softmax(logits, dim=1)

        # -------------- CLASSIFICATION-ONLY ---------------------------
        else:
            output = F.log_softmax(feats, dim=1)

        # -------------------------- LOSS ------------------------------
        if target is not None:
            ce = F.nll_loss(output, target)
            if self.use_slot:
                total = ce + self.lambda_value * attn_loss
                return output, [total, ce, attn_loss]
            return output, [ce]

        return output