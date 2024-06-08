#  Torch imports
import torch
torch.manual_seed(3407)
torch.cuda.manual_seed(3407)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
import deepspeed
cudnn.benchmark = True


from collections import defaultdict

# Python imports
import tqdm
from tqdm import tqdm
import os
from os.path import join as ospj
import csv

#Local imports
from data.meta_dataset import MetaDataset
from models.common import Classification
from utils.utils import save_args, load_args, load_args_test
from flags import parser, DATA_FOLDER
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from transformers.integrations import HfDeepSpeedConfig


from models.llamp import LLaMP

from transformers import LlamaForCausalLM, LlamaTokenizer

import json
import deepspeed
import numpy as np

def main():
    # Get arguments and start logging
    local_parser = deepspeed.add_config_arguments(parser)
    local_parser.add_argument('--target_dataset', type=str)
    args = local_parser.parse_args()
    logpath = args.logpath
    config = os.path.join(logpath, 'args_all.yaml') 

    import copy
    defaults = copy.deepcopy(args)
    
    # load_args_test(config, args)

    for key in defaults.__dict__:
        if defaults.__dict__[key] != args.__dict__[key]:
            print(f'{key}: {defaults.__dict__[key]}, {args.__dict__[key]}')


    device = torch.device(args.device)

    with open(args.deepspeed_config, 'r') as fp:
        deepspeed_config = json.load(fp) 

    dschf = HfDeepSpeedConfig(deepspeed_config)

    llama_model = LlamaForCausalLM.from_pretrained(args.model_base, device_map='cpu')
    tokenizer = LlamaTokenizer.from_pretrained(args.model_base)

    dataset = args.dataset

    results = defaultdict(list)

    base_testset = MetaDataset(
        phase='val',
        dataset=dataset,
        num_shots=args.coop_num_shots,
        seed=args.coop_seed,
    )

    new_testset = MetaDataset(
        phase='test',
        dataset=dataset,
        num_shots=args.coop_num_shots,
        seed=args.coop_seed,
    )

    classnames = {
        'base': base_testset.classnames,
        'new': new_testset.classnames,
    }
    model = LLaMP(base_testset, classnames, args, llama_model, tokenizer, few_shot=False)
        
    for i in range(1, 4):

        try:
            args.load = ospj(logpath,'ckpt_remodel_trimmed_{}.t7'.format(i))
            state_dict = torch.load(args.load, map_location='cpu')
        except:
            print("Failed to load model from checkpoint...")
        
        model.load_state_dict(state_dict, strict=False)

        model_engine, _, _, _ = deepspeed.initialize(config=deepspeed_config,
                                            model=model
                                        )
        model_engine.eval()

        base_testloader = torch.utils.data.DataLoader(
            base_testset,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=args.workers)

        new_testloader = torch.utils.data.DataLoader(
            new_testset,
            batch_size=args.test_batch_size,
            shuffle=False,
            num_workers=args.workers)
        
        evaluator_base = Classification(args, base_testset.idx2label)
        evaluator_new = Classification(args, new_testset.idx2label)

        with torch.no_grad(): 
            base_acc = test(0, model_engine, base_testloader, evaluator_base, args, logpath, device, subset='Base')['accuracy']
            new_acc = test(0, model_engine, new_testloader, evaluator_new, args, logpath, device, subset='New')['accuracy']

            hm = 2 * base_acc * new_acc / (base_acc + new_acc)

            print(f'Base: {base_acc}, New: {new_acc}, HM: {hm}')                    

            results[dataset].append(hm)

def test(epoch, model, testloader, evaluator, args, logpath, device, subset):
    '''
    Runs testing for an epoch
    '''

    evaluator.reset()
    model.eval()
    model.module.compute_all_class_embeddings(subset=subset.lower())

    for idx, data in tqdm(enumerate(testloader), total=len(testloader), desc='Testing on {}'.format(subset)):
        data  = [d.to(device) for d in data]
        data[0] = data[0].bfloat16()
        data[1] = data[1].bfloat16()

        with torch.inference_mode():
            _, predictions = model(data, subset=subset.lower())
        
        predictions = predictions.cpu()
        evaluator.process(predictions, data[-1].cpu())
    print("Done Running Results")
    stats = evaluator.evaluate()

    stats['a_epoch'] = epoch

    result = ''
    # write to Tensorboard
    for key in stats:
        result = result + key + '  ' + str(round(stats[key], 4)) + '| '

    result = result + args.name
    print(f'Test Epoch: {epoch}')
    print(result)

    return stats


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
