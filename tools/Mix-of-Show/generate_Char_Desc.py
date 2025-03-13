import time
import os
import json
import imageio
import requests

import argparse
from tqdm import tqdm
from moviepy.editor import VideoFileClip
from movieseq import MovieSeq_Char
import shutil
import openai
import cv2
import pandas as pd
import json

def parse_args():
    parser = argparse.ArgumentParser(description='MovieDirector', formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        "--openaikey",
        type=str,
        required=True,
        help="user query",
    )
    parser.add_argument(
        "--huggingfacekey",
        type=str,
        required=True,
        help="user query",
    )
    parser.add_argument(
        "--Char_url",
        type=str,
        required=False,
        help="model",
    )

    args = parser.parse_args()
    return args


def generate(openaikey,huggingfacekey,Char_url):
    os.environ['OPENAI_API_KEY'] = openaikey
    openai.api_key = os.getenv("OPENAI_API_KEY")
    char_size = 128

    # Please provide Huggingface tokens to access speaker-identify model
    HF_TOKEN = huggingfacekey

    # os.makedirs(Desc_url, exist_ok=True)

    cut_clip = 4

    # movieseq = MovieSeq()
    movieseq = MovieSeq_Char()

    for char_name in os.listdir(Char_url):
        if char_name == ".DS_Store" or char_name == "._.DS_Store":
            continue
        
        photo_p = os.path.join(Char_url,char_name)
        # save_desc_p = os.path.join(Char_url,char_name)
        # os.makedirs(save_desc_p, exist_ok=True)
        
        for char_photo in os.listdir(photo_p):
            photo_ONE = os.path.join(photo_p,char_photo)
            photo_ONE_save = os.path.join(photo_p,char_photo.replace("png","txt").replace("jpg","txt"))
            print("processing the file {}".format(photo_ONE_save))
            if "png" not in char_photo and "jpg" not in char_photo :
                continue

            image = cv2.imread(photo_ONE)

            height, width = image.shape[:2]

            # Determine the scale factor based on the shorter side
            if height < width:
                new_height = char_size
                new_width = int(width * (char_size / height))
            else:
                new_width = char_size
                new_height = int(height * (char_size / width))

            # Resize the image with the calculated dimensions
            resized_image = cv2.resize(image, (new_width, new_height))
            cv2.imwrite("./test.jpg", resized_image)

            query = """
                Based on the provided character names and stills for the Character, provide a concise description for the character appearance, within 20 words.
                The description must begin with the given character's name.
                Do not modify the provided name, including removing Spaces, etc.
                for example:  <The given character name> wears a red and black suit with a mask, featuring two swords on his back.
                """
            text = movieseq.get_response(char_name,"./test.jpg",query)
            text = text.replace(char_name,"<TOK>")
            
            with open(photo_ONE_save, 'w') as file:
                file.write(text)

            # print(text)
            # break
        # break


def main():
    args = parse_args()
    generate(args.openaikey,args.huggingfacekey,args.Char_url)



if __name__ == "__main__":
    main()
