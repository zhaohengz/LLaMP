declare -A datasets=(
    ['Caltech101']='Caltech101'
    ['FGVCAircraft']='FGVCAircraft'
    ['EuroSAT']='EuroSAT'
    ['ImageNet']='ImageNet'
    ['StanfordCars']='StanfordCars'
    ['DescribableTextures']='DescribableTextures'
    ['Food101']='Food101'
    ['OxfordPets']='OxfordPets'
    ['OxfordFlowers']='OxfordFlowers'
    ['SUN397']='SUN397'
    ['UCF101']='UCF101'
    ['ImageNetR']='ImageNetR'
    ['ImageNetA']='ImageNetA'
)

for dataset in "${!datasets[@]}"; do
    # use this command for few-shot
    # python tools/run_inference_chat_all.py --dataset "${datasets[$dataset]}" 
    python tools/run_inference_chat.py --dataset "${datasets[$dataset]}"
    python tools/extract_clip_text_features_zs.py --dataset "${datasets[$dataset]}"
done