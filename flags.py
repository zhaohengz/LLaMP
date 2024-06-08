import argparse

DATA_FOLDER = "./all_data"
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--config', default='configs/args.yml', help='path of the config file (training only)')
parser.add_argument('--dataset', default='ImageNet', help='')
parser.add_argument('--data_dir', default='', help='local path to data root dir from ' + DATA_FOLDER)
parser.add_argument('--logpath', default=None, help='Path to dir where to logs are stored (test only)')
parser.add_argument('--cv_dir', default='logs/', help='dir to save checkpoints and logs to')
parser.add_argument('--name', default='temp', help='Name of exp used to name models')
parser.add_argument('--load', default=None, help='path to checkpoint to load from')
parser.add_argument('--test_batch_size', type=int, default=16, help="Batch size at test/eval time")

parser.add_argument('--topk', type=int, default=1,help="Compute topk accuracy")
parser.add_argument('--workers', type=int, default=8,help="Number of workers")
parser.add_argument('--batch_size', type=int, default=4,help="Training batch size")
parser.add_argument('--lr', type=float, default=1e-4,help="Learning rate")
parser.add_argument('--lrg', type=float, default=1e-3,help="Learning rate feature extractor")
parser.add_argument('--weight_decay', type=float, default=5e-5,help="Weight decay")
parser.add_argument('--eval_val_every', type=int, default=1,help="Frequency of eval in epochs")
parser.add_argument('--max_epochs', type=int, default=20,help="Max number of epochs")

parser.add_argument('--device', default='cuda')
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
parser.add_argument('--distributed', action='store_true')

parser.add_argument("--model_base", type=str, default='meta-llama/Llama-2-7b-chat-hf')

parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument('--lr_scheduler', type=str, default=None)
parser.add_argument('--text_init', type=str, default='clip')
parser.add_argument('--lora_rank', default=8, type=int)
parser.add_argument('--lora_alpha', default=16, type=int)
parser.add_argument('--apply_primitive_loss', action="store_true")
parser.add_argument('--lora_on_vit', action="store_true")
parser.add_argument('--lora_dropout', default=0.1, type=float)
parser.add_argument('--pre_generated_caption', action="store_true")

parser.add_argument('--num_decoder_layers', default=1, type=int)
parser.add_argument('--prompt_learning', action="store_true")
parser.add_argument('--lora_decoding', action="store_true")
parser.add_argument('--freeze_vit', action="store_true")
parser.add_argument('--freeze_decoder_kv_proj', action="store_true")
parser.add_argument('--freeze_decoder_q_proj', action="store_true")
parser.add_argument('--freeze_decoder_o_proj', action="store_true")
parser.add_argument('--freeze_decoder_attn', action="store_true")
parser.add_argument('--freeze_decoder_ffn', action="store_true")

parser.add_argument('--distillation_type', default=None, type=str, help="")
parser.add_argument('--distillation_alpha', default=0.5, type=float, help="")
parser.add_argument('--distillation_tau', default=1.0, type=float, help="")
parser.add_argument('--distillation_lambda_fd', default=2000, type=float, help="")

parser.add_argument('--learn_class_embed_weight', action="store_true")

parser.add_argument('--coop_seed', default=1, type=int)
parser.add_argument('--coop_num_shots', default=16, type=int)

parser.add_argument('--test_per_class_result', action="store_true")
parser.add_argument('--test_compute_cmat', action="store_true")

parser.add_argument('--label_smoothing', default=0.1, type=float)

parser.add_argument('--num_llm_prompts', default=16, type=int)
parser.add_argument('--num_prior_tokens', default=100, type=int)

parser.add_argument('--visual_prompting', action="store_true")
parser.add_argument('--token_bias', action="store_true")
parser.add_argument('--prompt_dropout', default=0.0, type=float)
parser.add_argument('--prompt_type', type=str, default='suffix')

parser.add_argument('--num_text_ctx', type=int, default=4)
parser.add_argument('--num_vis_ctx', type=int, default=4)

parser.add_argument('--lambda_dist', type=float, default=1.0)
parser.add_argument('--llm_prompt_depth', type=int, default=9)
parser.add_argument('--lora_lr', default=2e-5, type=float)
parser.add_argument('--v_lora_start', default=6, type=int)
parser.add_argument('--v_lora_end', default=12, type=int)
parser.add_argument('--naive_decoding', action="store_true")
parser.add_argument('--past_key_value_file', default='release_past_key_value.pt', type=str)
parser.add_argument('--num_text_template', type=int, default=11)
parser.add_argument('--rand_aug', action="store_true")
parser.add_argument('--visual_prompt_depth', default=6, type=int)
parser.add_argument('--text_prompt_depth', default=9, type=int)
parser.add_argument('--prompt_combination', default='seq', type=str)
parser.add_argument('--clip_text_embed_file', default='release_clip_text_embeddings.pt', type=str)
parser.add_argument('--save_last', action="store_true")
parser.add_argument('--betas', nargs=2, type=float, default=[0.9, 0.999])

parser.add_argument('--debug', action="store_true")
parser.add_argument('--decoder_skip_connection', action="store_true")
parser.add_argument('--concat_fixed_prompts', action="store_true")

parser.add_argument("--img_dropout", default=0.0, type=float)