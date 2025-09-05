import torch
from model import HTR_VT


def main():
    # Basic shape test
    model = HTR_VT.create_model(nb_cls=80, img_size=[512, 64], encoder_type='conformer')
    model.eval()
    x = torch.randn(2, 1, 64, 512)
    with torch.no_grad():
        y = model(x)
    print('Input:', x.shape)
    print('Output:', y.shape)
    # CTC expects (B, T, C) or (T, B, C) after permute. Here y is (B, T, C) where T=sequence length.
    assert y.dim() == 3, 'Output must be 3D (B, T, C)'
    B, T, C = y.shape
    assert B == 2 and C == 80, 'Unexpected batch or class size'
    # Check no NaNs
    assert torch.isfinite(y).all(), 'Found non-finite values in output'
    print('Sanity check passed.')

    # Compare with ViT encoder quickly
    vit_model = HTR_VT.create_model(nb_cls=80, img_size=[512, 64], encoder_type='vit')
    vit_model.eval()
    with torch.no_grad():
        y_vit = vit_model(x)
    print('ViT output:', y_vit.shape)
    assert y_vit.shape == y.shape, 'Sequence shape mismatch between conformer and vit'
    print('Comparison passed.')

if __name__ == '__main__':
    main()
