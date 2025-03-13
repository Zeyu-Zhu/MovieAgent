import json
import os

import torch
from diffusers import DPMSolverMultistepScheduler

from mixofshow.pipelines.pipeline_edlora import EDLoRAPipeline

# 1004_Juno： Bren_MacGuff+Chin+Guy_Lab_Partner+Mac_MacGuff+Paulie_Bleeker+Leah+Mark_Loring+Girl_Lab_Partner+Juno_MacGuff+Vanessa_Loring+Liberty_Bell+Rollo
# 1005_Signs: Bren_MacGuff+Chin+Guy_Lab_Partner+Mac_MacGuff+Paulie_Bleeker+Leah+Mark_Loring+Girl_Lab_Partner+Juno_MacGuff+Vanessa_Loring+Liberty_Bell+Rollo
pretrained_model_path = 'experiments/composed_edlora/chilloutmix/NovelStory_1/combined_model_base'
enable_edlora = True  # True for edlora, False for lora

pipe = EDLoRAPipeline.from_pretrained(pretrained_model_path, scheduler=DPMSolverMultistepScheduler.from_pretrained(pretrained_model_path, subfolder='scheduler'), torch_dtype=torch.float16).to('cuda')
with open(f'{pretrained_model_path}/new_concept_cfg.json', 'r') as fr:
    new_concept_cfg = json.load(fr)
pipe.set_new_concept_cfg(new_concept_cfg)



prompt = "a <IronMan1> <IronMan2> emerges from the shadows, his eyes glowing with malevolence."
negative_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
print(prompt)
image = pipe(prompt, negative_prompt=negative_prompt, height=512, width=512, num_inference_steps=50, generator=torch.Generator('cuda').manual_seed(1), guidance_scale=7.5).images[0]

output_dir_name = "./IronMan.jpg"
image.save(output_dir_name)


prompt = "a <JakeSully1> <JakeSully2>"
negative_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
print(prompt)
image = pipe(prompt, negative_prompt=negative_prompt, height=512, width=512, num_inference_steps=50, generator=torch.Generator('cuda').manual_seed(1), guidance_scale=7.5).images[0]

output_dir_name = "./JakeSully.jpg"
image.save(output_dir_name)


# prompt = "a <Onetwo1> <Onetwo2>"
# negative_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
# print(prompt)
# image = pipe(prompt, negative_prompt=negative_prompt, height=512, width=512, num_inference_steps=50, generator=torch.Generator('cuda').manual_seed(1), guidance_scale=7.5).images[0]

# output_dir_name = "./Onetwo1.jpg"
# image.save(output_dir_name)


# prompt = "a <Disgust1> <Disgust2>"
# negative_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
# print(prompt)
# image = pipe(prompt, negative_prompt=negative_prompt, height=512, width=512, num_inference_steps=50, generator=torch.Generator('cuda').manual_seed(1), guidance_scale=7.5).images[0]

# output_dir_name = "./Disgust.jpg"
# image.save(output_dir_name)



# prompt = "a <Envy1> <Envy2>"
# negative_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
# print(prompt)
# image = pipe(prompt, negative_prompt=negative_prompt, height=512, width=512, num_inference_steps=50, generator=torch.Generator('cuda').manual_seed(1), guidance_scale=7.5).images[0]

# output_dir_name = "./Envy.jpg"
# image.save(output_dir_name)


# prompt = "a <Lijing1> <Lijing2>"
# negative_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'
# print(prompt)
# image = pipe(prompt, negative_prompt=negative_prompt, height=512, width=512, num_inference_steps=50, generator=torch.Generator('cuda').manual_seed(1), guidance_scale=7.5).images[0]

# output_dir_name = "./Lijing.jpg"
# image.save(output_dir_name)
