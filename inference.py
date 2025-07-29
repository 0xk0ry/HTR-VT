import torch
import os
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import utils
from data import dataset
from model import HTR_VT
from collections import OrderedDict

def load_model(args, device):
    model = HTR_VT.create_model(nb_cls=args.nb_cls, img_size=args.img_size[::-1])
    pth_path = os.path.join(args.save_dir, 'best_CER.pth')
    ckpt = torch.load(pth_path, map_location='cpu')
    model_dict = OrderedDict()
    pattern = re.compile('module.')

    for k, v in ckpt['state_dict_ema'].items():
        if re.search("module", k):
            model_dict[re.sub(pattern, '', k)] = v
        else:
            model_dict[k] = v

    model.load_state_dict(model_dict, strict=True)
    model = model.to(device)
    model.eval()
    return model

def llm_score(text, tokenizer, llm):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = llm(**inputs, labels=inputs["input_ids"])
    return -outputs.loss.item()

def infer(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(args, device)

    # Load LLM
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    llm = AutoModelForCausalLM.from_pretrained("gpt2")
    llm.eval()

    # Load test data
    test_dataset = dataset.myLoadDS(args.test_data_list, args.data_path, args.img_size)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.val_bs,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=args.num_workers)

    converter = utils.CTCLabelConverter(test_dataset.ralph.values())

    results = []
    for images, _ in test_loader:
        images = images.to(device)
        with torch.no_grad():
            logits = model(images)

        # Decode using CTC
        ctc_decoder = utils.CTCDecoder(converter)
        candidates = ctc_decoder.decode(logits)

        # Rescore with LLM
        rescored = []
        for seq, ctc_score in candidates:
            score = args.alpha * ctc_score + args.beta * llm_score(seq, tokenizer, llm)
            rescored.append((seq, score))

        best_seq = max(rescored, key=lambda x: x[1])[0]
        results.append(best_seq)

    return results

if __name__ == "__main__":
    from utils import option
    args = option.get_args_parser()
    print("Starting inference with arguments:", args)
    # results = infer(args)
    # for result in results:
    #     print(result)
