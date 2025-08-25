import torch

import os
import re
import json
import valid
from utils import utils
from utils import option
from data import dataset
from model import HTR_VT
from collections import OrderedDict


def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    args.save_dir = os.path.join(args.out_dir, args.exp_name)
    os.makedirs(args.save_dir, exist_ok=True)
    logger = utils.get_logger(args.save_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

    # Initialize model (tone head aware)
    if getattr(args, 'use_tone_head', False):
        base_charset_str = utils.build_base_charset()
        nb_cls = len(base_charset_str) + 1
        model = HTR_VT.create_model(nb_cls=nb_cls, img_size=args.img_size[::-1], use_tone_head=True)
        logger.info(f"Testing with tone head mode, base charset size: {nb_cls}")
    else:
        model = HTR_VT.create_model(nb_cls=args.nb_cls, img_size=args.img_size[::-1])
        logger.info(f"Testing with single head mode, charset size: {args.nb_cls}")

    pth_path = '/kaggle/input/htr-vt/pytorch/default-iam/8/checkpoint_0.0324_0.1066_100000.pth'
    logger.info('loading HWR checkpoint from {}'.format(pth_path))

    ckpt = torch.load(pth_path, map_location='cpu', weights_only=False)
    model_dict = OrderedDict()
    pattern = re.compile('module.')

    # Prefer 'state_dict_ema' if present else 'model'
    state_key = 'state_dict_ema' if 'state_dict_ema' in ckpt else ('model' if 'model' in ckpt else None)
    if state_key is None:
        raise KeyError("Checkpoint missing both 'state_dict_ema' and 'model' keys")

    for k, v in ckpt[state_key].items():
        if re.search("module", k):
            model_dict[re.sub(pattern, '', k)] = v
        else:
            model_dict[k] = v

    model.load_state_dict(model_dict, strict=True)
    model = model.cuda()

    logger.info('Loading test loader...')
    train_dataset = dataset.myLoadDS(args.train_data_list, args.data_path, args.img_size)

    test_dataset = dataset.myLoadDS(args.test_data_list, args.data_path, args.img_size, ralph=train_dataset.ralph)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=args.val_bs,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=args.num_workers)

    if getattr(args, 'use_tone_head', False):
        converter = utils.ToneLabelConverter(utils.build_base_charset())
    else:
        converter = utils.CTCLabelConverter(train_dataset.ralph.values())
    criterion = torch.nn.CTCLoss(reduction='none', zero_infinity=True).to(device)

    model.eval()
    with torch.no_grad():
        val_loss, val_cer, val_wer, preds, labels = valid.validation(model,
                                                                     criterion,
                                                                     test_loader,
                                                                     converter)

    logger.info(
        f'Test. loss : {val_loss:0.3f} \t CER : {val_cer:0.4f} \t WER : {val_wer:0.4f} ')


if __name__ == '__main__':
    args = option.get_args_parser()
    main()

