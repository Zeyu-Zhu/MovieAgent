import os
import json
import re

def modify_and_save_json_files(input_dir, output_dir,movie_name = "0001_American_Beauty"):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Loop through all files in the input directory
    for filename in os.listdir(input_dir):
        # Check if the file is a .json file
        if filename.endswith(".json"):
            # Get the name from the filename (before the extension)
            base_name = os.path.splitext(filename)[0]
            
            # Extract the character name from the file (after the last underscore)
            # Assuming the name is after the hyphen
            match = re.match(r".*-(.*)\.json", filename)
            if match:
                character_name = match.group(1)
            else:
                continue
            character_name_patg = filename
            # movie_name = "0001_American_Beauty"
            # Prepare the text by replacing 'Barbara_Fitts' with the character_name
            template = f"""name: {character_name}
movie: {movie_name}
manual_seed: 0
mixed_precision: fp16
gradient_accumulation_steps: 1

# dataset and data loader settings
datasets:
  train:
    name: LoraDataset
    concept_list: datasets/data_cfgs/MixofShow/single-concept/MovieGen/{movie_name}/{character_name_patg}
    use_caption: true
    use_mask: true
    instance_transform:
      - {{ type: HumanResizeCropFinalV3, size: 512, crop_p: 0.5 }}
      - {{ type: ToTensor }}
      - {{ type: Normalize, mean: [ 0.5 ], std: [ 0.5 ] }}
      - {{ type: ShuffleCaption, keep_token_num: 1 }}
      - {{ type: EnhanceText, enhance_type: human }}
    replace_mapping:
      <TOK>: <{character_name}1> <{character_name}2>
    batch_size_per_gpu: 2
    dataset_enlarge_ratio: 500

  val_vis:
    name: PromptDataset
    prompts: datasets/validation_prompts/single-concept/characters/test_man.txt
    num_samples_per_prompt: 1
    latent_size: [ 4,64,64 ]
    replace_mapping:
      <TOK>: <{character_name}1> <{character_name}2>
    batch_size_per_gpu: 4

models:
  pretrained_path: experiments/pretrained_models/chilloutmix
  enable_edlora: true  # true means ED-LoRA, false means vallina LoRA
  finetune_cfg:
    text_embedding:
      enable_tuning: true
      lr: !!float 1e-3
    text_encoder:
      enable_tuning: true
      lora_cfg:
        rank: 4
        alpha: 1.0
        where: CLIPAttention
      lr: !!float 1e-5
    unet:
      enable_tuning: true
      lora_cfg:
        rank: 4
        alpha: 1.0
        where: Attention
      lr: !!float 1e-4
  new_concept_token: <{character_name}1>+<{character_name}2>
  initializer_token: <rand-0.013>+man
  noise_offset: 0.01
  attn_reg_weight: 0.01
  reg_full_identity: false
  use_mask_loss: false
  gradient_checkpoint: false
  enable_xformers: true

# path
path:
  pretrain_network: ~

# training settings
train:
  optim_g:
    type: AdamW
    lr: !!float 0.0 # no use since we define different component lr in model
    weight_decay: 0.01
    betas: [ 0.9, 0.999 ] # align with taming

  # dropkv
  unet_kv_drop_rate: 0
  scheduler: linear
  emb_norm_threshold: !!float 5.5e-1

# validation settings
val:
  val_during_save: true
  compose_visualize: true
  alpha_list: [0, 0.7, 1.0] # 0 means only visualize embedding (without lora weight)
  sample:
    num_inference_steps: 50
    guidance_scale: 7.5

# logging settings
logger:
  print_freq: 10
  save_checkpoint_freq: !!float 10000

"""

            # Output file path
            output_file_path = os.path.join(output_dir, filename.replace("json","yml"))
            
            # Write the modified content to the new file
            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                output_file.write(template)

# Define the paths
input_directory = '/storage/wuweijia/MovieGen/Mix-of-Show/datasets/data_cfgs/MixofShow/single-concept/MovieGen/1054_Harry_Potter_and_the_prisoner_of_azkaban/'
output_directory = '/storage/wuweijia/MovieGen/Mix-of-Show/options/train/EDLoRA/MovieGen/1054_Harry_Potter_and_the_prisoner_of_azkaban/'
movie_name = "1054_Harry_Potter_and_the_prisoner_of_azkaban"

# Call the function
modify_and_save_json_files(input_directory, output_directory, movie_name)
