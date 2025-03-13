import os
import json
import argparse
import re


def parse_args():
    parser = argparse.ArgumentParser(description='MovieDirector', formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        "--directory_path",
        type=str,
        required=True,
        help="user query",
    )
    parser.add_argument(
        "--output_json_path",
        type=str,
        required=True,
        help="user query",
    )
    parser.add_argument(
        "--output_config",
        type=str,
        required=True,
        help="user query",
    )
    parser.add_argument(
        "--movie_name",
        type=str,
        required=True,
        help="user query",
    )
    

    args = parser.parse_args()
    return args



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
            # print(filename)
            character_name = filename.split(".")[0]
            
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


def create_json_for_files(directory, output_json):
    
    if not os.path.exists(output_json):
        os.makedirs(output_json)
        print(f"Directory '{output_json}' created.")

    # Traverse the directory
    for foldername in os.listdir(directory):
        if foldername == ".DS_Store":
            continue
        folder_path = os.path.join(directory, foldername)

        data_list = []
        # Check if it is a directory
        if os.path.isdir(folder_path):
            # Prepare the instance data structure
            instance_data = {
                "instance_prompt": "<TOK>",
                "instance_data_dir": folder_path,
                "caption_dir": folder_path.replace("Char_Bank", "Char_Desc_AutoStory")  # Replace part of the path for caption_dir
            }

            # Append the instance data to the list
            data_list.append(instance_data)

        # Write the list to the JSON file
        print(os.path.join(output_json, foldername+".json"))
        with open(os.path.join(output_json, foldername+".json"), 'w', encoding='utf-8') as json_file:
            json.dump(data_list, json_file, indent=4, ensure_ascii=False)

# Define the path to the directory and the output JSON file
# directory_path = '/storage/wuweijia/MovieGen/lsmdc/GT/Character_Bank/Char_Bank/Test/1054_Harry_Potter_and_the_prisoner_of_azkaban'
# output_json_path = '/storage/wuweijia/MovieGen/Mix-of-Show/datasets/data_cfgs/MixofShow/single-concept/MovieGen/1054_Harry_Potter_and_the_prisoner_of_azkaban/'


def main():
    args = parse_args()
    # Call the function to create the JSON
    create_json_for_files(args.directory_path, args.output_json_path)

    # Call the function
    modify_and_save_json_files(args.output_json_path, args.output_config, args.movie_name)



if __name__ == "__main__":
    main()
