# Large Language Models are Good Prompt Learners for Low-Shot Image Classification [CVPR 2024]

> [**Large Language Models are Good Prompt Learners for Low-Shot Image Classification**](https://arxiv.org/abs/2305.16681)<br>
> [Zhaoheng Zheng](https://zhaohengz.github.io/), [Jingmin Wei](https://github.com/Weijingmin2000), [Xuefeng Hu](https://xuefenghu.me/), [Haidong Zhu](https://haidongz.github.io) and [Ram Nevatia](https://sites.usc.edu/iris-cvlab/professor-ram-nevatia/)

Official implementation of [Large Language Models are Good Prompt Learners for Low-Shot Image Classification](https://arxiv.org/abs/2312.040761).


## Installation
We build our model based on `Python 3.11` and `PyTorch 2.2.0`. To prepare the environment, please follow the instructions below.

- Create a conda environment and install the requirements:
	```
	conda create -n llamp python=3.11 pip
	```
- Enter the environment:
	```
	conda activate llamp
	```
- Install the requirements:
	```
	pip install -r requirements.txt
	```
- Install `DASSL` from [this repo](https://github.com/KaiyangZhou/Dassl.pytorch)

## Datasets
Please follow [this link](https://github.com/muzairkhattak/PromptSRC/blob/main/docs/DATASETS.md) to prepare the datasets. The datasets should be organized as follows:
```
$DATA/
├── imagenet/
├── caltech-101/
├── oxford_pets/
├── stanford_cars/
...
```

After downloading the data, set the `DATA_FOLDER` variable in `flags.py` to your data path.

For LLaMA-2 weights, please visit [this link](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) to obtain the access directly from Meta.

### Preprocessing

You can download preprocessed metadata from [here](https://drive.google.com/drive/folders/16BE8Ns05mfLtI5Mv7tbu7LAMYD6HhUjK?usp=sharing) or run the following command to preprocess the data:
```
PYTHONPATH='.' tools/run_feature_extraction_all.sh
```

After you obtain the preprocessed metadata, please organize them as follows:
```
$DATA/
├── imagenet/
│   ├── release_past_key_value.pt
│   ├── release_clip_text_embeddings.pt
├── caltech-101/
│   ├── release_past_key_value.pt
│   ├── release_clip_text_embeddings.pt
...
```

## Checkpoints

We provide LLaMP checkpoints of all 11 datasets for the base-to-novel generalization benchmark. They can be downloaded from [here](https://drive.google.com/drive/folders/16BE8Ns05mfLtI5Mv7tbu7LAMYD6HhUjK?usp=sharing). After downloading the checkpoints, please organize them as follows:
```
checkpoints/
├── imagenet/
│   ├── release
│   |   ├── *.t7
├── caltech-101/
├── oxford_pets/
├── stanford_cars/
...
```


## Evaluation
To evaluate the model, run the following command:
```
 CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=False deepspeed test_llamp.py --deepspeed_config deepspeed_config/zero2_a100_40g.json --naive_decoding --freeze_decoder_kv --freeze_decoder_ffn --visual_prompting --dataset $DATASET --logpath $LOGPATH
 ```

, where `$DATASET` is the dataset name and `$LOGPATH` is the path where checkpoints are saved. 

`$DATASET` should be one of the following: `ImageNet`, `Caltech101`, `OxfordPets`, `StanfordCars`, `FGVCAircraft`, `OxfordFlowers`, `DescribableTextures`, `Food101`, `SUN397`, `UCF101`, `EuroSAT`.


## Training
Please run
```
bash scripts/launch/launch.sh $DATASET $SEED
```
to launch training. `$DATASET` is the dataset name and `$SEED` is the random seed chosen from 1, 2 and 3. 

`$DATASET` should be one of the following: `ImageNet`, `Caltech101`, `OxfordPets`, `StanfordCars`, `FGVCAircraft`, `OxfordFlowers`, `DescribableTextures`, `Food101`, `SUN397`, `UCF101`, `EuroSAT`.

# Citing LLaMP
If you find LLaMP useful in your research, please consider citing:
```
@InProceedings{Zheng_2024_Large,
  	title={Large Language Models are Good Prompt Learners for Low-Shot Image Classification},
  	author={Zheng, Zhaoheng and Wei, Jingmin and Hu, Xuefeng and Zhu, Haidong and Nevatia, Ram},
    	booktitle = {CVPR},
    	year      = {2024},
}
```
