import torch
import torch.nn as nn
import torch.nn.functional as F
from sloter.utils.slot_attention import SlotAttention
from sloter.utils.position_encode import build_position_encoding
from timm.models import create_model
from collections import OrderedDict

class Identical(nn.Module):
    def forward(self, x):      # pylint: disable=arguments-differ
        return x

def load_backbone(args):
    """timm backbone + 분류헤드 제거"""
    net = create_model(args.model,
                       pretrained=args.pre_trained,
                       num_classes=args.num_classes)

    # MNIST 흑백 지원
    if args.dataset == "MNIST":
        net.conv1 = nn.Conv2d(1, 64, 3, stride=2, padding=1, bias=False)

    # ── 분류 헤드 떼기 (slot 사용 시) ───────────────────────────────────────────
    if args.use_slot and not args.grad:
        if 'seresnet' in args.model:
            net.avg_pool, net.last_linear = Identical(), Identical()
        elif 'res' in args.model:
            net.global_pool, net.fc = Identical(), Identical()
        elif 'efficient' in args.model:
            net.global_pool, net.classifier = Identical(), Identical()
        elif 'densenet' in args.model:
            net.global_pool, net.classifier = Identical(), Identical()
        elif 'mobilenet' in args.model:
            net.global_pool = Identical()
            net.conv_head = Identical()
            net.act2 = Identical()
            net.classifier = Identical()

    # ── no-slot 사전학습 파라미터(optional) ───────────────────────────────────
    if args.use_slot and args.use_pre:
        ckpt = torch.load(f"saved_model/{args.dataset}_no_slot_checkpoint.pth",
                          map_location='cpu')
        new_state = {k[9:]: v for k, v in ckpt["model"].items()}   # remove 'backbone.'
        net.load_state_dict(new_state, strict=False)
        print("✔ no-slot backbone weights loaded")

    return net


class SlotModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.use_slot = args.use_slot
        self.backbone = load_backbone(args)

        if self.use_slot:
            # 백본 마지막 conv 채널 수 자동 추출(resnet/resnest/timm 계열 호환)
            self.backbone_out_dim = getattr(self.backbone, "num_features", None)
            if self.backbone_out_dim is None:                # timm 일부 모델 fallback
                self.backbone_out_dim = list(self.backbone.children())[-1].in_channels

            self.conv1x1 = nn.Conv2d(self.backbone_out_dim,
                                     args.hidden_dim,
                                     kernel_size=1,
                                     bias=False)

            # slot attention 모듈
            self.slot = SlotAttention(
                num_classes     = args.num_classes,
                slots_per_class = args.slots_per_class,
                dim             = args.hidden_dim,
                vis             = args.vis,
                vis_id          = args.vis_id,
                loss_status     = args.loss_status,
                power           = args.power,
                to_k_layer      = args.to_k_layer)

            self.position_emb = build_position_encoding('sine',
                                                        hidden_dim=args.hidden_dim)
            self.lambda_value = float(args.lambda_value)

            if args.pre_trained:
                self._freeze_layers(self.backbone, args.freeze_layers)
        elif args.pre_trained:
            self._freeze_layers(self.backbone, args.freeze_layers)

    @staticmethod
    def _freeze_layers(model, freeze_layer_num: int):
        """앞쪽 layer1·2·3·4 중 freeze_layer_num 만큼 동결"""
        if freeze_layer_num == 0:
            return
        unfreeze = ['layer4', 'layer3', 'layer2', 'layer1'][:4 - freeze_layer_num]
        for name, child in model.named_children():
            if any(layer in name for layer in unfreeze):
                continue
            for p in child.parameters():
                p.requires_grad = False
            SlotModel._freeze_layers(child, freeze_layer_num)

    def forward(self, x, target=None):
        # ① 4-D feature-map 확보 ---------------------------------------------
        if hasattr(self.backbone, "forward_features"):
            feat = self.backbone.forward_features(x)   # (B,C,H,W)
        else:                                          # odd custom backbone
            feat = self.backbone(x)                    # hope it's 4-D

        # ② 이하 동일 ---------------------------------------------------------
        if self.use_slot:
            feat = F.relu(self.conv1x1(feat))          # (B,D,H,W)
            feat_pe = feat + self.position_emb(feat)

            B, D, H, W = feat.shape
            tokens    = feat   .flatten(2).transpose(1, 2)  # (B,H*W,D)
            tokens_pe = feat_pe.flatten(2).transpose(1, 2)

            logits, attn_loss = self.slot(tokens_pe, tokens)
        else:
            logits = feat.mean(dim=[2, 3])             # GAP

        logp = F.log_softmax(logits, dim=1)

        # ── 훈련
        if target is not None:
            ce_loss = F.nll_loss(logp, target)
            if self.use_slot:
                loss = ce_loss + self.lambda_value * attn_loss
                return logp, [loss, ce_loss, attn_loss]
            return logp, [ce_loss]

        # ── 추론
        return logp
