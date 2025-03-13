from PIL import Image, ImageDraw, ImageFont
import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
import os
import cv2
import time
from pathlib import Path
from loguru import logger
from datetime import datetime

from .hyvideo.utils.file_utils import save_videos_grid
from .hyvideo.config import parse_args
from .hyvideo.inference import HunyuanVideoSampler


class HunyuanVideo_I2V_pipe:
    def __init__(self,args):

        # print(args)
        # args = parse_args(args)

        models_root_path = Path(args.model_base)
        if not models_root_path.exists():
            raise ValueError(f"`models_root` not exists: {models_root_path}")

        # Load models
        self.hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)

        # Get the updated args
        args = self.hunyuan_video_sampler.args
        self.args = args

    def predict(self, prompt, image_path, video_save_path, size = (569, 320)):
        
        self.args.prompt = prompt
        self.args.video_size[0] = size[1]
        self.args.video_size[1] = size[0]
        self.args.video_length = 129
        self.args.i2v_resolution = "540p"
        self.args.i2v_image_path = image_path



        outputs = self.hunyuan_video_sampler.predict(
        prompt=self.args.prompt, 
        height=self.args.video_size[0],
        width=self.args.video_size[1],
        video_length=self.args.video_length,
        seed=self.args.seed,
        negative_prompt=self.args.neg_prompt,
        infer_steps=self.args.infer_steps,
        guidance_scale=self.args.cfg_scale,
        num_videos_per_prompt=self.args.num_videos,
        flow_shift=self.args.flow_shift,
        batch_size=self.args.batch_size,
        embedded_guidance_scale=self.args.embedded_cfg_scale,
        i2v_mode=self.args.i2v_mode,
        i2v_resolution=self.args.i2v_resolution,
        i2v_image_path=self.args.i2v_image_path,
        i2v_condition_type=self.args.i2v_condition_type,
        i2v_stability=self.args.i2v_stability
        )
        samples = outputs['samples']

        for i, sample in enumerate(samples):
            sample = samples[i].unsqueeze(0)
            # time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
            # cur_save_path = f"{save_path}/{time_flag}_seed{outputs['seeds'][i]}_{outputs['prompts'][i][:100].replace('/','')}.mp4"
            save_videos_grid(sample, video_save_path, fps=24)
            # logger.info(f'Sample save to: {video_save_path}')



