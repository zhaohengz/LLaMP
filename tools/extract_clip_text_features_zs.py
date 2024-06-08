from argparse import Namespace, ArgumentParser
from data.meta_dataset import MetaDataset
import os
from transformers import CLIPConfig, CLIPModel, CLIPProcessor
from flags import DATA_FOLDER
from utils.utils import clean_text
from tools.gpt_templates import GPT_TEMPLATES, CUSTOM_IMAGENET_TEMPLATES
import torch
import spacy
from tqdm import tqdm

import torch.nn.functional as F

CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}

cmd_parser = ArgumentParser()
cmd_parser.add_argument('--dataset', type=str, required=True)

cmd_args = cmd_parser.parse_args()

args = Namespace(
    dataset=cmd_args.dataset,
    clip_config='clip-vit-base-patch16',
    seed=1,
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

clip_model = CLIPModel.from_pretrained("openai/{}".format(args.clip_config)).cuda()
processor = CLIPProcessor.from_pretrained("openai/{}".format(args.clip_config))


output_embeddings = {}

data_dir = datasets['base'].data_dir

cur_tempalte = CUSTOM_TEMPLATES[args.dataset]


content_dict = torch.load(os.path.join(DATA_FOLDER, data_dir, '0427_test_7B_GPTAug_NounPhrase_full_past_key_value.pt'))
nlp_engine = spacy.load("en_core_web_sm")


for split in ['base', 'new']:
    dataset = datasets[split]
    all_text_embeddings = []
    for temp in tqdm(GPT_TEMPLATES[args.dataset] + CUSTOM_IMAGENET_TEMPLATES):
        temp_text_embeddings = []
        for i, c in enumerate(dataset.classnames):
            noun_chunks = []
            for content in content_dict[split][:1]:
                doc = nlp_engine(content['text_predictions'][i])
                noun_chunks += list(doc.noun_chunks)
            base_text = CUSTOM_TEMPLATES[args.dataset].format(c.replace("_", " ")) + 'with '

            all_text_embeds = []

            with torch.inference_mode():
                for n in noun_chunks:
                    full_text = base_text + n.text
                    full_input = processor(full_text, return_tensors="pt", padding=True, max_length=77, truncation=True)
                    full_input = {k: v.cuda() for k, v in full_input.items()}
                    text_embeds = F.normalize(clip_model.get_text_features(**full_input), dim=-1).cuda()
                    all_text_embeds.append(text_embeds)

            temp_text_embeddings.append(torch.cat(all_text_embeds, dim=0).mean(dim=0))
            
        all_text_embeddings.append(torch.stack(temp_text_embeddings, dim=0))

    output_embeddings[split] = {
        'avg': torch.stack(all_text_embeddings, dim=0).mean(dim=0),
        'all': torch.stack(all_text_embeddings, dim=0)
    }

torch.save(output_embeddings, os.path.join(DATA_FOLDER, data_dir, 'clip_text_embeddings.pt'))