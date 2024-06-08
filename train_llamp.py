#  Torch imports
import torch
torch.manual_seed(3407)
torch.cuda.manual_seed(3407)
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import deepspeed
cudnn.benchmark = True

# Python imports
import tqdm
from tqdm import tqdm
import os
from os.path import join as ospj
import csv

#Local imports
from data.meta_dataset import MetaDataset
from models.common import Classification
from utils.utils import save_args, load_args
from flags import parser, DATA_FOLDER
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW, SGD

from utils.utils import init_distributed_mode, is_main_process

from transformers import get_scheduler
from transformers.integrations import HfDeepSpeedConfig


from models.llamp import LLaMP
from torch.optim.lr_scheduler import LambdaLR
import math

from transformers import AutoModelForCausalLM, AutoTokenizer

from decimal import Decimal

import json

def get_cosine_schedule_lambda_with_warmup(
    min_lr, base_lr, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return max(min_lr / base_lr, float(current_step) / float(max(1, num_warmup_steps)))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr / base_lr, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return lr_lambda

def get_cosine_schedule_with_warmup(
    optimizer, min_lrs, base_lrs, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = []
    for min_lr, base_lr in zip(min_lrs, base_lrs):
        lr_lambda.append(get_cosine_schedule_lambda_with_warmup(min_lr, base_lr, num_warmup_steps, num_training_steps, num_cycles))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def save_checkpoint(model, epoch, logpath, seed, filename):
    state = {
        'net': model.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, os.path.join(logpath, 'ckpt_{}_{}.t7'.format(filename, seed)))

def build_optimizer_parameters(config, model):

    param_optimizer = list(model.named_parameters())
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0] and 'lora' not in n[0] and 'visual_projection' not in n[0]]
    lora_params = [n for n in model.named_parameters() if 'lora' in n[0] or 'visual_projection' in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight', 'pos_embed','relative_position_bias_table']

    weight_decay = getattr(config, "weight_decay", 0.01)
    optimizer_grouped_parameters = [{
        'params': [
            p for n, p in param_optimizer
            if p.requires_grad
        ],
        'weight_decay':
        weight_decay
    }, {
        'params': [p for n, p in lora_params if p.requires_grad],
        'weight_decay': weight_decay,
        'lr': config.lora_lr
    }
    ]
    
    return optimizer_grouped_parameters

def main():
    # Get arguments and start logging
    local_parser = deepspeed.add_config_arguments(parser)
    args = local_parser.parse_args()

    device = torch.device(args.device)

    load_args(args.config, args)


    project_name_suffix = "-{class_emb}-{lr:.1E}-{lr_scheuler}-{prompt_type}-{num_prior_tokens}Pr{llm_prompt_depth}x{num_llm_prompts}P{num_text_ctx}T{num_vis_ctx}V-{bias}Init-{num_template}xTemplate-{dist_type}{lambda_dist:.1f}xDist{randaug}-{betas}-WD{wd}".format(
        class_emb='{:d}X{}'.format(args.num_decoder_layers, "decode"),
        lr=Decimal(args.lr),
        lr_scheuler=args.lr_scheduler if args.lr_scheduler else "constant",
        prompt_type=args.prompt_type,
        num_prior_tokens=args.num_prior_tokens,
        llm_prompt_depth=args.llm_prompt_depth,
        num_llm_prompts=args.num_llm_prompts,
        num_text_ctx=args.num_text_ctx,
        num_vis_ctx=args.num_vis_ctx,
        dist_type=args.distillation_type,
        lambda_dist=args.lambda_dist,
        bias='Bias' if args.token_bias else 'No',
        num_template=args.num_text_template,
        randaug='-RandAug' if args.rand_aug else '',
        betas='-'.join([str(b) for b in args.betas]),
        wd=args.weight_decay,
    )

    project_name_suffix  = project_name_suffix + '-Skip' if args.decoder_skip_connection else project_name_suffix
    project_name_suffix  = project_name_suffix + '-ConcatPrior' if args.concat_fixed_prompts else project_name_suffix

    args.name = args.name + project_name_suffix 

    logpath = os.path.join(args.cv_dir, args.name)
    if is_main_process():
        os.makedirs(logpath, exist_ok=True)
        save_args(args, logpath, args.config)

    with open(args.deepspeed_config, 'r') as fp:
        deepspeed_config = json.load(fp) 

    dschf = HfDeepSpeedConfig(deepspeed_config)

    # init deepspeed

    model = AutoModelForCausalLM.from_pretrained(args.model_base, device_map='cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.model_base)

    # Get dataset
    trainset = MetaDataset(
        phase='train',
        dataset=args.dataset,
        num_shots=args.coop_num_shots,
        seed=args.coop_seed,
        num_template=args.num_text_template,
        rand_aug=args.rand_aug
    )

    base_testset = MetaDataset(
        phase='val',
        dataset=args.dataset,
        num_shots=args.coop_num_shots,
        seed=args.coop_seed,
    )

    new_testset = MetaDataset(
        phase='test',
        dataset=args.dataset,
        num_shots=args.coop_num_shots,
        seed=args.coop_seed,
    )

    classnames = {
        'base': trainset.classnames,
        'new': new_testset.classnames,
    }

    model = LLaMP(trainset, classnames, args, model, tokenizer)

    start_epoch = 0
    # Load checkpoint
    if args.load is not None:
        checkpoint = torch.load(args.load)
        model.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        print('Loaded model from ', args.load)

    parameter_group = build_optimizer_parameters(args, model)

    optimizer = AdamW(
        lr=args.lr, betas=tuple(args.betas), weight_decay=args.weight_decay, params=parameter_group
    )

    model_engine, _, _, _ = deepspeed.initialize(config=deepspeed_config,
                                                model=model,
                                                optimizer=optimizer,
                                                model_parameters=parameter_group)
    
    sampler = DistributedSampler(trainset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True, drop_last=False, seed=3407)
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        sampler=sampler)

    if args.freeze_vit:
        min_lrs = [1e-5]
        base_lrs= [args.lr]
    else:
        min_lrs = [1e-5, max(args.lora_lr / args.lr * 1e-5, 5e-6)]
        base_lrs= [args.lr, args.lora_lr]

    if args.lr_scheduler:
        if args.lr_scheduler == 'cosine':
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                min_lrs=min_lrs,
                base_lrs=base_lrs,
                num_warmup_steps=1,
                num_training_steps=max(args.max_epochs, 20) + 1
            )
        else:
            lr_scheduler = get_scheduler(
                name=args.lr_scheduler,
                optimizer=optimizer,
                num_warmup_steps=0,
                num_training_steps=max(args.max_epochs, 20))
    else:
        lr_scheduler = None

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

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    tokenizer.padding_side = "left"  # Allow batched inference

    for epoch in tqdm(range(start_epoch, args.max_epochs), desc = 'Current epoch'):
        trainloader.sampler.set_epoch(epoch)
        train(epoch, model_engine, tokenizer, trainloader, False, lr_scheduler, device)
        if is_main_process():
            if (epoch + 1) % args.eval_val_every == 0:
                with torch.no_grad(): # todo: might not be needed
                    base_acc = test(epoch, model_engine, base_testloader, evaluator_base, args, logpath, device, subset='Base')['accuracy']
                    new_acc = test(epoch, model_engine, new_testloader, evaluator_new, args, logpath, device, subset='New')['accuracy']

                    hm = 2 * base_acc * new_acc / (base_acc + new_acc)

                    print(f'Base: {base_acc}, New: {new_acc}, HM: {hm}')                    


        dist.barrier()  

    if is_main_process():
        save_checkpoint(model_engine, epoch, logpath, args.coop_seed, 'last')

def train(epoch, model_engine, tokenizer, trainloader, ema, lr_scheduler, device):
    '''
    Runs training for an epoch
    '''

    model_engine.train() # Let's switch to training

    train_loss = {
        'loss_total': 0.0,
        'loss_ce': 0.0,
        'loss_l1': 0.0,
        'loss_dist': 0.0,
    } 
    for idx, data in tqdm(enumerate(trainloader), total=len(trainloader), desc = 'Training'):
        # text_inputs = tokenizer(data[-1], padding='longest', truncation=True, max_length=50, return_tensors="pt") 
        # data[-1] = text_inputs

        # if idx == 0:
        #     print(data[1:], lr_scheduler.get_lr())

        if epoch == 0 and idx == len(trainloader) // 10:
            lr_scheduler.step()

        data  = [d.to(device) for d in data]
        data[0] = data[0].bfloat16()
        data[1] = data[1].bfloat16()
        
        losses, _ = model_engine(data)

        loss = losses['loss_total']

        model_engine.backward(loss)
        model_engine.step()

        if ema:
            model_engine.module.update()
            
        for key in train_loss:
            train_loss[key] += losses[key].item()

    if lr_scheduler is not None:
        lr_scheduler.step()

    train_loss = {k: v/len(trainloader) for k, v in train_loss.items()}
    print('Epoch: {}| Loss: {}'.format(epoch, round(train_loss['loss_total'], 2)))


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
