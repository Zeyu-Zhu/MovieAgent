1-3步骤准备数据训练single lora for each character.
4-6步骤训练得到最终的多人 lora
7- 步骤获取layout

export PATH="/users/wuweijia/anaconda3/bin:$PATH"


0. 生成每个character的描述
python generate_Char_Desc.py


1. Convert the placeholder of description of each character to <TOK>. (将character描述里面的character name替换为<TOK>)

python Step_1_convert_TOK_for_character.py

2. Generate the data .json file for each Character. (为每个character写一个data json文件，方便训对应的lora)

python Step_2_generate_data_json_for_character.py

3. Generate the config .json file for each Character. (为每个character写一个config json文件，方便训对应的lora)

python Step_3_generate_config_json_for_character.py


4. train lora for each character

export PYTHONPATH=/storage/wuweijia/MovieGen/Mix-of-Show:$PYTHONPATH 

for file in /storage/wuweijia/MovieGen/Mix-of-Show/options/train/EDLoRA/MovieGen/1041_This_is_40/*.yml; do
    CUDA_VISIBLE_DEVICES=4 accelerate launch train_edlora.py -opt "$file"
done


5. Generate the config .json file for multi lora.

python Step_4_generate_config_json_for_multi_lora.py

6. train multi lora

sh train_lora_multi.sh

7. generate the layout for each shot video. (MovieSeq)

python generate_box_for_AutoStory.py

8. generate single pose for each character. 

CUDA_VISIBLE_DEVICES=1 python sample_single_for_fusion.py 

cd Grounded-Segment-Anything (AutoStory环境)
bash GroundingSam.sh

cd mmpose  
bash run_mmpose_batch.sh   (AutoStory环境)

or 
cd pytorch-openpose    (mmpose)
python demo.py

python story_utils/compose_keypose.py --ann /storage/wuweijia/MovieGen/Mix-of-Show/datasets/layout/1054_Harry_Potter_and_the_prisoner_of_azkaban --pose /storage/wuweijia/MovieGen/Mix-of-Show/datasets/single_person/1054_Harry_Potter_and_the_prisoner_of_azkaban --save_path /storage/wuweijia/MovieGen/Mix-of-Show/datasets/multi_pose/1054_Harry_Potter_and_the_prisoner_of_azkaban

9. final
bash infer_movie.sh

10. Using image-video generation model (SVD)
cd SVD
CUDA_VISIBLE_DEVICES=4 python scripts/sampling/simple_video_sample.py








