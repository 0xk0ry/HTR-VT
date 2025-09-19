import argparse
import os
import sys

# Ensure workspace root is on sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Adjust imports to your package layout (model_v1)
from model_v1.model import HTR_VT


def log_param_breakdown(module, title="Model", max_levels=(1, 2)):
    params = list(module.named_parameters())
    total = sum(p.numel() for _, p in params)
    trainable = sum(p.numel() for _, p in params if p.requires_grad)
    print(f"[PARAM] {title}: total={total:,} trainable={trainable:,}")

    from collections import defaultdict
    def group_by(depth):
        grp = defaultdict(int)
        grp_tr = defaultdict(int)
        for name, p in params:
            parts = name.split('.')
            key = '.'.join(parts[:depth]) if parts else '(root)'
            grp[key] += p.numel()
            if p.requires_grad:
                grp_tr[key] += p.numel()
        items = sorted(grp.items(), key=lambda kv: kv[1], reverse=True)
        return items, grp_tr

    for depth in max_levels:
        items, grp_tr = group_by(depth)
        print(f"[PARAM] {title} by {depth}-level group (top 20):")
        for k, v in items[:20]:
            tr = grp_tr.get(k, 0)
            print(f"  - {k:<40} total={v:,} trainable={tr:,}")


def count_params_of(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nb-cls', type=int, default=80)
    parser.add_argument('--img-h', type=int, default=64)
    parser.add_argument('--img-w', type=int, default=512)
    args = parser.parse_args()

    model = HTR_VT.create_model(nb_cls=args.nb_cls, img_size=(args.img_w, args.img_h))

    # Overall
    log_param_breakdown(model, title="Base Model")

    # Split counts: ResNet backbone vs Transformer encoder
    # From model_v1.model.HTR_VT.MaskedAutoencoderViT:
    #  - ResNet backbone is `patch_embed = resnet18.ResNet18(embed_dim)`
    #  - Transformer encoder is `blocks` (ModuleList of Block) plus `norm` and attention/MLP within blocks
    # We'll report:
    #   ResNet params = patch_embed
    #   Encoder params = blocks + norm
    #   Head params = head (classifier)
    #   Positional/mask token and layer_norm counted separately for completeness

    resnet = getattr(model, 'patch_embed', None)
    blocks = getattr(model, 'blocks', None)
    enc_norm = getattr(model, 'norm', None)
    head = getattr(model, 'head', None)
    pos_embed = getattr(model, 'pos_embed', None)
    mask_token = getattr(model, 'mask_token', None)
    pre_layer_norm = getattr(model, 'layer_norm', None)

    res_total = res_train = enc_total = enc_train = 0
    head_total = head_train = other_total = other_train = 0

    if resnet is not None:
        res_total, res_train = count_params_of(resnet)
    if blocks is not None:
        btot, btr = count_params_of(blocks)
        enc_total += btot
        enc_train += btr
    if enc_norm is not None:
        nt, ntr = count_params_of(enc_norm)
        enc_total += nt
        enc_train += ntr
    if head is not None:
        head_total, head_train = count_params_of(head)
    # small tensors
    if pos_embed is not None:
        t = pos_embed.numel(); tr = pos_embed.numel() if pos_embed.requires_grad else 0
        other_total += t; other_train += tr
    if mask_token is not None:
        t = mask_token.numel(); tr = mask_token.numel() if mask_token.requires_grad else 0
        other_total += t; other_train += tr
    if pre_layer_norm is not None:
        t, tr = count_params_of(pre_layer_norm)
        other_total += t; other_train += tr

    print("\n[PARAM] Split summary:")
    print(f"  - ResNet backbone             total={res_total:,} trainable={res_train:,}")
    print(f"  - Transformer encoder         total={enc_total:,} trainable={enc_train:,}")
    print(f"  - Classification head         total={head_total:,} trainable={head_train:,}")
    print(f"  - Other (pos/mask/pre-norm)   total={other_total:,} trainable={other_train:,}")


if __name__ == '__main__':
    main()
