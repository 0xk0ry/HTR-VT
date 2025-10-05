import argparse
import os
import sys
from collections import defaultdict
import json

ROOT = os.path.abspath(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from model.HTR_VT import create_model  # noqa: E402
import torch  # noqa: E402


def fmt(n):
    return f"{n:,}"


def count(p):
    t = sum(x.numel() for x in p)
    tr = sum(x.numel() for x in p if x.requires_grad)
    return t, tr


def add_group(stats, name, params):
    t, tr = count(params)
    stats.append((name, t, tr))


def print_block_line(name, t, tr, width=38):
    print(f"  - {name:<{width}} total={fmt(t):>10} trainable={fmt(tr):>10}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nb-cls", type=int, default=80)
    ap.add_argument("--img-w", type=int, default=512)
    ap.add_argument("--img-h", type=int, default=64)
    ap.add_argument("--embed-dim", type=int, default=768)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--num-heads", type=int, default=6)
    ap.add_argument("--dummy-forward", action="store_true",
                    help="Run a dummy forward to instantiate lazy modules (e.g., proj_in).")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--mlp-ratio", type=float, default=4.0)
    ap.add_argument("--van-detail", action="store_true", help="Print internal breakdown of each VAN block (grouped by immediate submodule).")
    ap.add_argument("--json-out", type=str, default=None, help="Optional path to save JSON summary.")
    args = ap.parse_args()

    model = create_model(
        nb_cls=args.nb_cls,
        img_size=(args.img_w, args.img_h),
    ).to(args.device)

    # Force lazy projection creation (proj_in) if requested
    if args.dummy_forward:
        with torch.no_grad():
            dummy = torch.randn(1, 3, args.img_h, args.img_w, device=args.device)
            model.forward(dummy, use_masking=False)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[PARAM] Model total={fmt(total_params)} trainable={fmt(trainable_params)}")

    stats = []

    # Major components
    add_group(stats, "patch_embed (ResNet18)", model.patch_embed.parameters())
    if hasattr(model, "proj_in") and model.proj_in is not None:
        add_group(stats, "proj_in (1x1 align)", model.proj_in.parameters())
    add_group(stats, "van_reducer", model.van_reducer.parameters())
    add_group(stats, "transformer.blocks", model.blocks.parameters())
    add_group(stats, "transformer.norm", model.norm.parameters())
    add_group(stats, "head", model.head.parameters())
    add_group(stats, "layer_norm (post-head)", model.layer_norm.parameters())
    # Positional and mask token (pos_embed is frozen)
    add_group(stats, "mask_token", [model.mask_token])
    add_group(stats, "pos_embed (frozen)", [model.pos_embed])

    print("\n[PARAM] Component summary:")
    for name, t, tr in stats:
        print_block_line(name, t, tr)

    # VAN block breakdown
    print("\n[PARAM] VAN blocks detail:")
    van_block_details = []
    if hasattr(model.van_reducer, "blocks"):
        for i, blk in enumerate(model.van_reducer.blocks):
            bt, btr = count(blk.parameters())
            print_block_line(f"van_reducer.blocks[{i}]", bt, btr)
            if args.van_detail:
                # Group parameters by immediate submodule prefix
                subgroups = defaultdict(lambda: [0, 0])
                for name, p in blk.named_parameters():
                    top = name.split('.')[0]
                    subgroups[top][0] += p.numel()
                    if p.requires_grad:
                        subgroups[top][1] += p.numel()
                for sg, (st, str_) in sorted(subgroups.items(), key=lambda kv: kv[1][0], reverse=True):
                    print_block_line(f"    {sg}", st, str_)
                van_block_details.append({
                    'index': i,
                    'total': bt,
                    'trainable': btr,
                    'submodules': {k: {'total': v[0], 'trainable': v[1]} for k, v in subgroups.items()}
                })
            else:
                van_block_details.append({'index': i, 'total': bt, 'trainable': btr})

    # Transformer block internal breakdown
    print("\n[PARAM] Transformer encoder blocks detail:")
    transformer_block_details = []
    for i, blk in enumerate(model.blocks):
        blk_params = list(blk.parameters())
        bt, btr = count(blk_params)
        print_block_line(f"blocks[{i}]", bt, btr)
        # Internals
        # Attention pieces
        qkv_t, qkv_tr = count([blk.attn.qkv.weight, *( [blk.attn.qkv.bias] if blk.attn.qkv.bias is not None else [] )])
        proj_t, proj_tr = count([blk.attn.proj.weight, *( [blk.attn.proj.bias] if blk.attn.proj.bias is not None else [] )])
        # MLP pieces (timm Mlp: fc1, act, drop, fc2, drop)
        fc1 = [blk.mlp.fc1.weight, blk.mlp.fc1.bias]
        fc2 = [blk.mlp.fc2.weight, blk.mlp.fc2.bias]
        fc1_t, fc1_tr = count(fc1)
        fc2_t, fc2_tr = count(fc2)
        print_block_line(f"    attn.qkv", qkv_t, qkv_tr)
        print_block_line(f"    attn.proj", proj_t, proj_tr)
        print_block_line(f"    mlp.fc1", fc1_t, fc1_tr)
        print_block_line(f"    mlp.fc2", fc2_t, fc2_tr)
        transformer_block_details.append({
            'index': i,
            'total': bt,
            'trainable': btr,
            'attn_qkv': {'total': qkv_t, 'trainable': qkv_tr},
            'attn_proj': {'total': proj_t, 'trainable': proj_tr},
            'mlp_fc1': {'total': fc1_t, 'trainable': fc1_tr},
            'mlp_fc2': {'total': fc2_t, 'trainable': fc2_tr}
        })

    # Theoretical transformer block params (ignoring norms & biases)
    # (4 + 2 * mlp_ratio) * D^2
    D = args.embed_dim
    theoretical_block = int((4 + 2 * args.mlp_ratio) * (D ** 2))
    print(f"\n[INFO] Theoretical per transformer block (no norm/bias): {fmt(theoretical_block)}")
    if len(model.blocks):
        print(f"[INFO] First block actual (incl. norms/bias): {fmt(transformer_block_details[0]['total'])}")

    # Category level summary (Attention vs MLP vs VAN)
    attn_total = sum(d['attn_qkv']['total'] + d['attn_proj']['total'] for d in transformer_block_details)
    mlp_total = sum(d['mlp_fc1']['total'] + d['mlp_fc2']['total'] for d in transformer_block_details)
    van_total = sum(v['total'] for v in van_block_details)
    core_sum = attn_total + mlp_total + van_total
    def pct(x):
        return f"{(x / core_sum * 100):5.2f}%" if core_sum > 0 else "  n/a "
    print("\n[PARAM] Core component proportions (excl. patch_embed/head/pos):")
    print_block_line(f"Attention (qkv+proj) [{pct(attn_total)}]", attn_total, attn_total)
    print_block_line(f"MLP (fc1+fc2)        [{pct(mlp_total)}]", mlp_total, mlp_total)
    print_block_line(f"VAN blocks           [{pct(van_total)}]", van_total, van_total)

    # Aggregate by top-level prefix (optional)
    agg = defaultdict(lambda: [0, 0])
    for name, p in model.named_parameters():
        top = name.split('.')[0]
        agg[top][0] += p.numel()
        if p.requires_grad:
            agg[top][1] += p.numel()

    print("\n[PARAM] Top-level aggregation:")
    for k, (t, tr) in sorted(agg.items(), key=lambda kv: kv[1][0], reverse=True):
        print_block_line(k, t, tr)

    # Optional JSON export
    if args.json_out:
        summary = {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'component_summary': [
                {'name': n, 'total': t, 'trainable': tr} for (n, t, tr) in stats
            ],
            'van_blocks': van_block_details,
            'transformer_blocks': transformer_block_details,
            'theoretical_transformer_block': theoretical_block,
            'category_proportions': {
                'attention_total': attn_total,
                'mlp_total': mlp_total,
                'van_total': van_total
            },
            'top_level': {k: {'total': v[0], 'trainable': v[1]} for k, v in agg.items()}
        }
        with open(args.json_out, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\n[INFO] JSON summary written to {args.json_out}")

    print("\nDone.")


if __name__ == "__main__":
    main()