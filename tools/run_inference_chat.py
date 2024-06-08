from argparse import Namespace, ArgumentParser
import torch
import torch.nn.functional as F
from data.meta_dataset import MetaDataset
import os
from flags import DATA_FOLDER
from utils.utils import clean_text

from tools.gpt_llama_templates import GPT_LLAMA_TEMPLATES

from transformers import AutoTokenizer, AutoModelForCausalLM

from tqdm import tqdm

import spacy

CUSTOM_TEMPLATES = {
    "OxfordPets": "the distinctive appearance of a {}, a type of pet.",
    "OxfordFlowers": "the distinctive appearance of a {}, a type of flower.",
    "FGVCAircraft": "the distinctive appearance of a {}, a type of aircraft.",
    "DescribableTextures": "the distinctive appearance of {} texture.",
    "EuroSAT": "the distinctive appearance of a centered satellite photo of {}.",
    "StanfordCars": "the distinctive appearance of a {}.",
    "Food101": "the distinctive appearance of {}, a type of food.",
    "SUN397": "the distinctive appearance of a {}.",
    "Caltech101": "the distinctive appearance of a {}.",
    "UCF101": "the distinctive appearance of a person doing {}.",
    "ImageNet": "the distinctive appearance of a {}.",
    "ImageNetSketch": "the distinctive appearance of a {}.",
    "ImageNetV2": "the distinctive appearance of a {}.",
    "ImageNetA": "the distinctive appearance of a {}.",
    "ImageNetR": "the distinctive appearance of a {}.",
}

def apply_chat_template(tokenizer, sentence):
    chat = [
        {"role": "system", "content": "You are a helpful assistant."},
    ]
    chat.append(
        {"role": "user", "content": sentence}
    )

    return tokenizer.apply_chat_template(chat, tokenize=False)

cmd_parser = ArgumentParser()
cmd_parser.add_argument('--dataset', type=str, required=True)

cmd_args = cmd_parser.parse_args()

args = Namespace(
    dataset=cmd_args.dataset,
    clip_config='clip-vit-base-patch16',
    seed=1,
    model_path="meta-llama/Llama-2-7b-chat-hf",
    model_base=None
)

datasets = {
	'base' : MetaDataset(
        dataset=args.dataset,
        phase='train',
        seed=args.seed,
        num_shots=1
    ),
    'new' : MetaDataset(
        dataset=args.dataset,
        phase='test',
        seed=args.seed,
        num_shots=1
    )
}

model = AutoModelForCausalLM.from_pretrained(args.model_path).cuda()
tokenizer = AutoTokenizer.from_pretrained(args.model_path)

nlp_engine = spacy.load("en_core_web_sm")

tokenizer.use_default_system_prompt = False

tokenizer.pad_token_id = (
    tokenizer.unk_token_id # unk. we want this to be different from the eos token
)
tokenizer.padding_side = "left"  # Allow batched inference

dump_dict = {}

temp = CUSTOM_TEMPLATES[args.dataset]

output_embeddings = {}

data_dir = datasets['base'].data_dir

model = model.cuda()

for target in ['base', 'new']:
    dump_dict[target] = []
    dataset = datasets[target]

    for gpt_template in GPT_LLAMA_TEMPLATES[args.dataset]:
        wordfill = [gpt_template.format(c.replace("_", " ")) for c in 
                 dataset.classnames]

        text_input = []
        for idx, p in enumerate(wordfill):
            text_input.append(apply_chat_template(tokenizer, 'In one sentece, ' + p))

        for idx, p in enumerate(dataset.classnames):
            text_input[idx] = text_input[idx] + " The {} appears as".format(p.replace('_', ' '))

        text_input = tokenizer(text_input, return_tensors='pt', padding=True, truncation=True, max_length=128) 

        batch_size = 16

        valid_pkv = []
        next_token_embeds_list = []
        next_token_attn_mask = []

        model.eval()

        text_predictions = []
        num_pair = text_input['input_ids'].shape[0]

        input_ids = text_input['input_ids'].cuda()
        attention_mask = text_input['attention_mask'].cuda()


        with torch.inference_mode():

            noun_phrase_inputs = []
            for i in tqdm(range(0, num_pair, batch_size)):

                num_input_tokens = input_ids[i:i+batch_size].shape[1]

                output_ids = model.generate(
                    input_ids=input_ids[i:i+batch_size],
                    attention_mask=attention_mask[i:i+batch_size],
                    do_sample=False,
                    max_new_tokens=64,
                    temperature=None,
                    top_p=None,
                    use_cache=True,
                )

                num_input_ids = input_ids[i:i+batch_size].shape[1]
                next_tokens = output_ids[:, num_input_ids:]

                out = tokenizer.batch_decode(output_ids[:, num_input_tokens:], skip_special_tokens=True)

                for o in out:
                    text_predictions.append(o)
                    doc = nlp_engine(o)
                    all_noun_phrases = ", ".join([chunk.text for chunk in doc.noun_chunks])
                    noun_phrase_inputs.append(all_noun_phrases)

            noun_phrase_inputs = tokenizer(noun_phrase_inputs, return_tensors='pt', padding=True, truncation=True, max_length=128)
            output_attn_mask = []

            for i in tqdm(range(0, num_pair, batch_size)):
                extended_model_inputs = model.prepare_inputs_for_generation(
                    input_ids=torch.cat([input_ids[i:i+batch_size], noun_phrase_inputs['input_ids'][i:i+batch_size].cuda()], dim=1),
                    attention_mask=torch.cat([attention_mask[i:i+batch_size], noun_phrase_inputs['attention_mask'][i:i+batch_size].cuda()], dim=1),
                    use_cache=True)
                                
                outputs = model(**extended_model_inputs, return_dict=True)

                pkv = torch.stack([torch.stack([kv.cpu() for kv in pkvs]) for pkvs in outputs.past_key_values[-1:]])
                valid_pkv.append(pkv)
                output_attn_mask.append(torch.cat([attention_mask[i:i+batch_size], noun_phrase_inputs['attention_mask'][i:i+batch_size].cuda()], dim=1).cpu())

            
        valid_pkv = torch.cat(valid_pkv, dim=2)
        output_attn_mask = torch.cat(output_attn_mask, dim=0)

        dump_dict[target].append({
            'past_key_values': valid_pkv, 
            'attn_mask': output_attn_mask, 
            'text_predictions': text_predictions
        })

    torch.save(dump_dict, os.path.join(DATA_FOLDER, data_dir, 'full_past_key_value.pt'))
