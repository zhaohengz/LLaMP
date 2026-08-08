

# Los Modelos de Lenguaje Grande son Buenos Aprendices de Prompt para la Clasificación de Imágenes con Pocos Datos [CVPR 2024]

> [**Los Modelos de Lenguaje Grande son Buenos Aprendices de Prompt para la Clasificación de Imágenes con Pocos Datos**](https://arxiv.org/abs/2312.04076)<br>
> [Zhaoheng Zheng](https://zhaohengz.github.io/), [Jingmin Wei](https://github.com/Weijingmin2000), [Xuefeng Hu](https://xuefenghu.me/), [Haidong Zhu](https://haidongz-usc.github.io/) y [Ram Nevatia](https://sites.usc.edu/iris-cvlab/professor-ram-nevatia/)

Implementación oficial de [Los Modelos de Lenguaje Grande son Buenos Aprendices de Prompt para la Clasificación de Imágenes con Pocos Datos](https://arxiv.org/abs/2312.04076).


## Instalación
Construimos nuestro modelo con `Python 3.11` y `PyTorch 2.2.0`. Para preparar el entorno, siga las instrucciones a continuación.

- Cree un entorno conda e instale los requisitos:
	```
	conda create -n llamp python=3.11 pip
	```
- Ingrese al entorno:
	```
	conda activate llamp
	```
- Instale los requisitos:
	```
	pip install -r requirements.txt
	```
- Instale `DASSL` desde [este repositorio](https://github.com/KaiyangZhou/Dassl.pytorch)

## Conjuntos de Datos
Siga [este enlace](https://github.com/muzairkhattak/PromptSRC/blob/main/docs/DATASETS.md) para preparar los conjuntos de datos. Los conjuntos de datos deben organizarse de la siguiente manera:
```
$DATA/
├── imagenet/
├── caltech-101/
├── oxford_pets/
├── stanford_cars/
...
```

Después de descargar los datos, establezca la variable `DATA_FOLDER` en `flags.py` a la ruta de sus datos.

Para los pesos de LLaMA-2, visite [este enlace](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) para obtener acceso directamente desde Meta.

### Preprocesamiento

Puede descargar los metadatos preprocesados desde [aquí](https://drive.google.com/drive/folders/16BE8Ns05mfLtI5Mv7tbu7LAMYD6HhUjK?usp=sharing) o ejecutar el siguiente comando para preprocesar los datos:
```
PYTHONPATH='.' tools/run_feature_extraction_all.sh
```

Una vez que obtenga los metadatos preprocesados, organícelos de la siguiente manera:
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

Proporcionamos checkpoints de LLaMP para los 11 conjuntos de datos del benchmark de generalización de base a novedad. Pueden descargarse desde [aquí](https://drive.google.com/drive/folders/16BE8Ns05mfLtI5Mv7tbu7LAMYD6HhUjK?usp=sharing). Después de descargar los checkpoints, organícelos de la siguiente manera:
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


## Evaluación
Para evaluar el modelo, ejecute el siguiente comando:
```
 CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=False deepspeed test_llamp.py --deepspeed_config deepspeed_config/zero2_a100_40g.json --naive_decoding --freeze_decoder_kv --freeze_decoder_ffn --visual_prompting --dataset $DATASET --logpath $LOGPATH
 ```

, donde `$DATASET` es el nombre del conjunto de datos y `$LOGPATH` es la ruta donde se guardan los checkpoints. 

`$DATASET` debe ser uno de los siguientes: `ImageNet`, `Caltech101`, `OxfordPets`, `StanfordCars`, `FGVCAircraft`, `OxfordFlowers`, `DescribableTextures`, `Food101`, `SUN397`, `UCF101`, `EuroSAT`.


## Entrenamiento
Por favor, ejecute
```
bash scripts/launch/launch.sh $DATASET $SEED
```
para iniciar el entrenamiento. `$DATASET` es el nombre del conjunto de datos y `$SEED` es la semilla aleatoria elegida entre 1, 2 y 3. 

`$DATASET` debe ser uno de los siguientes: `ImageNet`, `Caltech101`, `OxfordPets`, `StanfordCars`, `FGVCAircraft`, `OxfordFlowers`, `DescribableTextures`, `Food101`, `SUN397`, `UCF101`, `EuroSAT`.

# Citando a LLaMP
Si encuentra LLaMP útil en su investigación, considere citarlo:
```
@InProceedings{Zheng_2024_Large,
  	title={Large Language Models are Good Prompt Learners for Low-Shot Image Classification},
  	author={Zheng, Zhaoheng and Wei, Jingmin and Hu, Xuefeng and Zhu, Haidong and Nevatia, Ram},
    	booktitle = {CVPR},
    	year      = {2024},
}
```
