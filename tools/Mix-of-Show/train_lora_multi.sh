export PYTHONPATH=/storage/wuweijia/MovieGen/MovieDirector/MovieDirector/tools/Mix-of-Show:$PYTHONPATH 


config_file="Bajie+Wukong+ErLang"
save_dir="NovelStory_2"

CUDA_VISIBLE_DEVICES=6 python gradient_fusion.py \
    --concept_cfg="datasets/data_cfgs/MixofShow/multi-concept/MovieGen/NovelStory_2/${config_file}.json" \
    --save_path="experiments/composed_edlora/chilloutmix/${save_dir}" \
    --pretrained_models="experiments/pretrained_models/chilloutmix" \
    --optimize_textenc_iters=1000 \
    --optimize_unet_iters=100