import random
import numpy as np
import argparse
import datetime
import time
from pathlib import Path

import torch
from torch.utils.data import DistributedSampler

import tools.prepare_things as prt
from dataset.choose_dataset import select_dataset
from engine import train_one_epoch, evaluate
from sloter.slot_model import SlotModel
from tools.calculate_tool import MetricLog
from tools.prepare_things import DataLoaderX

###############################################################################
# Helper – hot‑patch SlotAttention to avoid GRU shape errors                  #
###############################################################################

def _patch_slot_attention():

    # Lazy import so that the module must exist in the runtime environment
    from sloter.utils.slot_attention import SlotAttention as _SA
    import torch

    def _forward(self, inputs, inputs_x):  # pylint: disable=too-many-locals
        """Re‑implementation of *forward* with safe GRU call."""
        B, N, D = inputs.shape                # (batch, N_inputs, dim)
        S       = self.num_slots

        # Initial slots -------------------------------------------------------
        slots = self.initial_slots.expand(B, -1, -1)                          # (B, S, D)

        # Pre‑compute keys / values ------------------------------------------
        k = self.to_k(inputs)                                                 # (B, N, D)
        v = inputs_x                                                         # (B, N, D)

        # Needed outside the loop for loss -----------------------------------
        attn = None

        for _ in range(self.iters):
            slots_prev = slots                                                # (B, S, D)

            # --- Attention --------------------------------------------------
            q     = slots * self.scale                                        # (B, S, D)
            logits = torch.einsum("bsd,bnd->bsn", q, k)                     # (B, S, N)
            attn   = logits.softmax(dim=-1) + self.eps                       # (B, S, N)

            updates = torch.einsum("bnd,bsn->bsd", v, attn) / N             # (B, S, D)

            # --- GRU expects (seq, batch, dim) ------------------------------
            self.gru.flatten_parameters()
            gru_in   = updates.reshape(1, B * S, D)                           # seq_len=1
            gru_state= slots_prev.reshape(1, B * S, D).contiguous()
            slots, _ = self.gru(gru_in, gru_state)                            # output: (1, B·S, D)
            slots    = slots.view(B, S, D)

        # Aggregate slots per class if required ------------------------------
        if self.slots_per_class > 1:
            updates_cls = updates.view(B, self.num_classes, self.slots_per_class, D).sum(2)
        else:
            updates_cls = updates                                             # (B, C, D)

        # Slot‑based auxiliary loss -----------------------------------------
        attn_relu = torch.relu(attn)
        slot_loss = attn_relu.sum() / (attn.numel())

        return self.loss_status * updates_cls.sum(dim=2), slot_loss.pow(self.power)

    # Apply monkey‑patch only once
    if not hasattr(_SA, "_patched"):
        _SA.forward = _forward            # type: ignore[attr-defined]
        _SA._patched = True

###############################################################################
# Argument parser                                                             #
###############################################################################

def get_args_parser():
    """Return the CLI argument parser."""

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in {"yes", "true", "t", "y", "1"}:
            return True
        if v.lower() in {"no", "false", "f", "n", "0"}:
            return False
        raise argparse.ArgumentTypeError("Unsupported value encountered.")

    parser = argparse.ArgumentParser("SCOUTER – Slot‑Attention trainer", add_help=False)

    # ── basic ---------------------------------------------------------------
    parser.add_argument("--model",         default="resnet18")
    parser.add_argument("--dataset",       default="MNIST")
    parser.add_argument("--dataset_dir",   default="./data/")
    parser.add_argument("--img_size",      default=260,    type=int)
    parser.add_argument("--output_dir",    default="saved_model/")
    parser.add_argument("--device",        default="cuda")
    parser.add_argument("--num_workers",   default=1,      type=int)

    # ── optimisation --------------------------------------------------------
    parser.add_argument("--lr",             default=1e-4, type=float)
    parser.add_argument("--slot_lr_mult",   default=5.0,  type=float, help="LR multiplier for slot / FC parameters")
    parser.add_argument('--lr_fc', default=None, type=float, help="(deprecated) absolute LR for slot / FC params")
    parser.add_argument("--weight_decay",   default=1e-4, type=float)
    parser.add_argument("--batch_size",     default=64,   type=int)
    parser.add_argument("--epochs",         default=20,   type=int)
    parser.add_argument("--lr_drop",        default=70,   type=int,
                        help="StepLR drop interval (epochs)")
# 👇 바로 아래에 추가
    parser.add_argument('--slot_iters', type=int, default=3,
                    help='number of recurrent iterations in Slot-Attention')

    # ── early‑stopping / checkpoint ----------------------------------------
    parser.add_argument("--early_stop_patience", default=5, type=int)
    parser.add_argument("--best_ckpt_name",      default="best_auc.pth")

    # ── SCOUTER / slot options ---------------------------------------------
    parser.add_argument("--num_classes",       default="10")
    parser.add_argument("--use_slot",          default=True,  type=str2bool)
    parser.add_argument("--freeze_layers",     default=0,     type=int)
    parser.add_argument("--lambda_value",      default="1.")
    parser.add_argument("--slots_per_class",   default="3")
    parser.add_argument("--power",             default="2")
    parser.add_argument("--to_k_layer",        default=1,     type=int)
    parser.add_argument("--loss_status",       default=1,     type=int)
    parser.add_argument("--hidden_dim",        default=64,    type=int)
    parser.add_argument("--channel",           default=512,   type=int)
    parser.add_argument("--use_pre",           default=False, type=str2bool)
    parser.add_argument("--pre_trained",       default=True,  type=str2bool)
    parser.add_argument("--vis",               default=False, type=str2bool)
    parser.add_argument("--vis_id",            default=0,     type=int)

    # ── misc ----------------------------------------------------------------
    parser.add_argument("--aug",               default=True,  type=str2bool)
    parser.add_argument("--aug_level",         default="base", choices=["base", "strong"])
    parser.add_argument("--grad",              default=False, type=str2bool)
    parser.add_argument("--grad_min_level",    default=0.,    type=float)
    parser.add_argument("--cal_area_size",     default=False, type=str2bool)
    parser.add_argument("--iterated_evaluation_num", default=1, type=int)
    parser.add_argument("--thop",              default=False, type=str2bool)

    # ── distributed ---------------------------------------------------------
    parser.add_argument("--world_size",  default=1, type=int)
    parser.add_argument("--local_rank",  type=int)
    parser.add_argument("--dist_url",    default="env://")
    parser.add_argument("--resume",      default="",  help="checkpoint path")
    parser.add_argument("--start_epoch", default=0,    type=int)

    return parser

###############################################################################
# Main training routine                                                       #
###############################################################################

def main(args):
    seed = getattr(args, "seed", 777) 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Patch SlotAttention at runtime ***before*** model construction
    _patch_slot_attention()

    # — distributed / device —
    prt.init_distributed_mode(args)
    device = torch.device(args.device)

    # — model —
    model = SlotModel(args).to(device)
    print("train model:", "use slot" if args.use_slot else "without slot",
          "negative loss" if args.use_slot and args.loss_status != 1 else "positive loss")

    model_wo_ddp = model

    if args.lr_fc is not None:
        # args.lr == backbone LR → multiplier = lr_fc / lr
        args.slot_lr_mult = args.lr_fc / args.lr
        print(f"▶ using lr_fc={args.lr_fc:.3g}  → slot_lr_mult={args.slot_lr_mult:.1f}")

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu] if hasattr(args, "gpu") else None,
            find_unused_parameters=True,
        )
        model_wo_ddp = model.module

    # — parameter groups & optimiser —
    try:
        param_groups = model_wo_ddp.param_groups(base_lr=args.lr,
                                                slot_lr_mult=args.slot_lr_mult)
    except AttributeError:
        # Fallback: simple 2‑group split
        slot_params, backbone_params = [], []
        for n, p in model_wo_ddp.named_parameters():
            if not p.requires_grad:
                continue
            if any(k in n for k in ["slot", "conv1x1", "backbone.fc"]):
                slot_params.append(p)
            else:
                backbone_params.append(p)
        param_groups = [
            {"params": backbone_params, "lr": args.lr},
            {"params": slot_params,     "lr": args.lr * args.slot_lr_mult},
        ]

    optimizer    = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_drop)

    # — datasets & loaders —
    ds_train, ds_val, _ = select_dataset(args)
    if args.distributed:
        sampler_train = DistributedSampler(ds_train)
        sampler_val   = DistributedSampler(ds_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(ds_train)
        sampler_val   = torch.utils.data.SequentialSampler(ds_val)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, args.batch_size, drop_last=True)
    loader_train = DataLoaderX(ds_train, batch_sampler=batch_sampler_train, num_workers=args.num_workers)
    loader_val   = DataLoaderX(ds_val, args.batch_size, sampler=sampler_val, num_workers=args.num_workers)

    # — optional resume —
    output_dir = Path(args.output_dir)
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        if args.use_slot:
            backbone_ckpt = {k: v for k, v in ckpt["model"].items() if k.startswith("backbone.")}
            model_wo_ddp.backbone.load_state_dict(backbone_ckpt, strict=False)
            print(f"▶ Loaded backbone from {args.resume}")
        else:
            model_wo_ddp.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
            args.start_epoch = ckpt["epoch"] + 1
            print(f"▶ Resumed full model from epoch {ckpt['epoch']}")

    # — log utils —
    print("Start training")
    start_time   = time.time()
    log          = MetricLog()
    record       = log.record
    best_auc     = -1.0
    patience_cnt = 0

    # — training loop —
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)

        train_one_epoch(model, loader_train, optimizer, device, record, epoch)
        lr_scheduler.step()

        # periodic checkpoint (lightweight)
        if prt.is_main_process():
            ckpt_path = output_dir / "latest.pth"
            torch.save({"model": model_wo_ddp.state_dict(), "epoch": epoch}, ckpt_path)

        # validation
        evaluate(model, loader_val, device, record, epoch)
        torch.cuda.empty_cache()
        log.print_metric()

        cur_auc = record["val"].get("auc", [None])[-1]
        if cur_auc is None:
            continue
        print(f"Epoch {epoch:02d} | Train Acc {record['train']['acc'][-1]:.3f} | "
              f"Val Acc {record['val']['acc'][-1]:.3f} | Val AUC {cur_auc:.4f}")

        if cur_auc > best_auc:
            best_auc = cur_auc
            patience_cnt = 0
            best_path = output_dir / args.best_ckpt_name
            prt.save_on_master({"model": model_wo_ddp.state_dict(), "auc": best_auc, "epoch": epoch}, best_path)
            print(f"  ▲ New best AUC! checkpoint saved to {best_path.name}")
        else:
            patience_cnt += 1
            print(f"  ▼ AUC not improved ({patience_cnt}/{args.early_stop_patience})")
            if patience_cnt >= args.early_stop_patience:
                print(f"Early stopping at epoch {epoch}. Best Val AUC = {best_auc:.4f}")
                break

    elapsed = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f"Training time {elapsed}")

###############################################################################
# Grid‑search helper                                                          #
###############################################################################

def param_translation(args):
    args_d      = vars(args)
    sweep_keys  = [
        ("num_classes",       int),
        ("lambda_value",      float),
        ("power",             int),
        ("slots_per_class",   int),
        ("hidden_dim",        int),
    ]

    # detect sweep target (first CLI arg containing comma‑separated list)
    target_key = None
    for k, tp in sweep_keys:
        val = args_d[k]
        if isinstance(val, str) and "," in val:
            target_key = k
            candidates = [tp(v) for v in val.split(",")]
            break
        args_d[k] = tp(val)

    if target_key is None:
        main(args)
        return

    all_results = {}
    for cand in candidates:
        args_d[target_key] = cand
        all_results[f"{target_key}-{cand}"] = [main(args) for _ in range(args.iterated_evaluation_num)]
    print(all_results)

###############################################################################
# Entry‑point                                                                 #
###############################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Training / evaluation script", parents=[get_args_parser()])
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    param_translation(args)
