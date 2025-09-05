import torch
from model_conformer.model import HTR_VT

"""Simple sanity script.
Runs a forward pass with shape prints enabled and optional masking.
Usage (PowerShell):
  python -m model_conformer.sanity_check
"""

def run():
    nb_cls = 80
    img_size = [512, 32]
    model = HTR_VT.create_model(nb_cls=nb_cls, img_size=img_size, encoder_type="conformer", depth=4, num_heads=6)
    model.eval()

    dummy = torch.randn(2, 1, img_size[0], img_size[1])
    with torch.no_grad():
        print("== Forward (debug) ==")
        _ = model.forward(dummy, mask_ratio=0.0, max_span_length=1, use_masking=False)  # prints due to debug path if set
        # second pass with masking example
        print("== Forward with masking (10% spans of length 2) ==")
        _ = model.forward(dummy, mask_ratio=0.1, max_span_length=2, use_masking=True)

    # CTC layout note
    out = model(dummy)
    print("Final logits shape:", out.shape, "(expect [B,T,C])")
    print("To feed into torch.nn.CTCLoss transpose to [T,B,C]:", out.permute(1,0,2).shape)

if __name__ == "__main__":
    run()
