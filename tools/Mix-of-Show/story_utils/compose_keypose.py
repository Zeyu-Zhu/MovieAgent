import os
import numpy as np
from PIL import Image
import cv2
import math
import argparse
from omegaconf import OmegaConf


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ann", type=str, help="path to compose config file", required=True)
    parser.add_argument("--pose", type=str, help="path to compose config file", required=True)
    parser.add_argument("--save_path", type=str, help="path to compose config file", required=True)
    # parser.add_argument("--config", type=str, help="path to compose config file", required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    for ann in os.listdir(args.ann):
        ann_path = os.path.join(args.ann,ann)
        arrr = args.pose.split("/")[-1]+"/"+ann.replace(".json","")
        pose_path = os.path.join(args.pose,ann.replace(".json","_grounding_dino_pose"))

        config = OmegaConf.load(ann_path)
        # height, width = 512, config.image_width

        output_dir = args.save_path
        os.makedirs(output_dir, exist_ok=True)

        width = 2048
        height = 1024
        img = np.zeros((height, width, 3), np.uint8)

        input_subject_dict = config["box"]
        
        for key_name in input_subject_dict:
            
            pose_path_one = os.path.join(pose_path,key_name.replace(" ","_")+"_pose.png")
            # subject_path_list = sub["keypose_path"]
            x0, y0, w, h  = input_subject_dict[key_name]
            x1 = x0 + w
            y1 = y0 + h
            subject_width = x1 - x0
            subject_height = y1 - y0
            # load image with PIL
            print(y0,y1, x0,x1)
            print(subject_width, subject_height)
            try:
                subject_img = Image.open(pose_path_one)
                subject_img = subject_img.resize((subject_width, subject_height))  # TODO we should keep aspect ratio
                subject_img = np.array(subject_img)
            except:
                continue
            
            # draw the subject
            try:
                img[y0:y1, x0:x1] = subject_img
            except:
                continue
        img = Image.fromarray(img)
        output_path = os.path.join(output_dir, ann.replace(".json",".png"))
        img.save(output_path)
        

        