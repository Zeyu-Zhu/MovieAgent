import os
from datetime import datetime
import argparse

from base_agent import BaseAgent
from system_prompts import sys_prompts
from tools import ToolCalling, save_json
import json
from moviepy.editor import VideoFileClip, concatenate_videoclips
from pathlib import Path
import yaml

def parse_args():
    
    # parser = argparse.ArgumentParser(description='MovieAgent', formatter_class=argparse.RawTextHelpFormatter)
    parser = argparse.ArgumentParser(description="MovieAgent")

    parser.add_argument(
        "--script_path",
        type=str,
        required=True,
        help="user query",
    )
    parser.add_argument(
        "--character_photo_path",
        type=str,
        required=True,
        help="user query",
    )
    parser.add_argument(
        "--LLM",
        type=str,
        required=False,
        help="model: gpt4-o | deepseek-r1 | deepseek-v3",
    )
    parser.add_argument(
        "--gen_model",
        type=str,
        required=False,
        help="model: ROICtrl | StoryDiffusion",
    )
    parser.add_argument(
        "--audio_model",
        type=str,
        required=False,
        help="model",
    ) 
    parser.add_argument(
        "--talk_model",
        type=str,
        required=False,
        help="model",
    )
    parser.add_argument(
        "--Image2Video",
        type=str,
        required=False,
        help="model: SVD | I2Vgen | CogVideoX",
    )

    args = parser.parse_args()

    if args.gen_model:
        config = load_config(args.gen_model)
        # print(config)
        for key, value in config.items():
            if not getattr(args, key, None):  
                setattr(args, key, value)
    
    if args.Image2Video:
        config = load_config(args.Image2Video)
        # print(config)
        for key, value in config.items():
            if not getattr(args, key, None):  
                setattr(args, key, value)

    return args


def load_config(model_name):
    """ with model_name, read config """
    config_path = Path(f"configs/{model_name}.json")  

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            if config_path.suffix == ".json":
                return json.load(f)
            elif config_path.suffix in [".yaml", ".yml"]:
                return yaml.safe_load(f)
    return {}  

class ScriptBreakAgent:
    def __init__(self, args, sample_model="sdxl-1", audio_model="VALL-E", talk_model = "Hallo2", Image2Video = "CogVideoX",
                 script_path = "", character_photo_path="", save_mode="img"):
        self.args = args
        self.sample_model = sample_model
        self.audio_model = audio_model
        self.talk_model = talk_model
        self.Image2Video = Image2Video
        self.script_path = script_path
        self.character_photo_path = character_photo_path
        self.characters_list = []
        self.movie_name = script_path.split("/")[-1].replace(".json","")
        self.save_mode = save_mode

        self.update_info()
        self.init_agent()
        self.init_videogen()
    
    def init_videogen(self):
        movie_script, characters_list = self.extract_characters_from_json(self.script_path, 40)

        self.tools = ToolCalling(self.args, sample_model=self.sample_model, audio_model = self.audio_model, \
                                 talk_model = self.talk_model, Image2Video = self.Image2Video, \
                                    photo_audio_path = self.character_photo_path, \
                                    characters_list=characters_list, save_mode=self.save_mode)
    
    def init_agent(self):
        # initialize agent
        self.screenwriter_agent = BaseAgent(self.args.LLM, system_prompt=sys_prompts["screenwriterCoT-sys"], use_history=False, temp=0.7)
        # self.supervisor_agent = BaseAgent(system_prompt=sys_prompts["scriptsupervisor-sys"], temp=0.7)

        self.sceneplanning_agent = BaseAgent(self.args.LLM, system_prompt=sys_prompts["ScenePlanningCoT-sys"], use_history=False, temp=0.7)

        self.shotplotcreate_agent = BaseAgent(self.args.LLM, system_prompt=sys_prompts["ShotPlotCreateCoT-sys"], use_history=False, temp=0.7)
        
        
        
    def format_results(self, results):
        formatted_text = "Observation:\n\n"
        for item in results:
            formatted_text += f"Prompt: {item['Prompt']}\n"
            for question, answer in zip(item["Questions"], item["Answers"]):
                formatted_text += f"Question: {question} -- Answer: {answer}\n"
            formatted_text += "\n"
        return formatted_text

    
    def update_info(self):
        folder_name = self.script_path.split("/")[-2]
        self.save_path = f"./Results/{folder_name}"
        
        model_config = self.args.LLM + "_" + self.sample_model + "_" + self.args.Image2Video 
        self.video_save_path = os.path.join(self.save_path, model_config, "video")

        self.sub_script_path = os.path.join(self.save_path, model_config, f"Step_1_script_results.json")
        self.scene_path = os.path.join(self.save_path, model_config, f"Step_2_scene_results.json")
        self.shot_path = os.path.join(self.save_path, model_config, f"Step_3_shot_results.json")

        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(self.video_save_path, exist_ok=True)

    def read_json(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data

    def extract_characters_from_json(self,file_path, n):
        data = self.read_json(file_path)
        movie_script = data['MovieScript']
        characters = data['Character']
        selected_characters = characters[:n]
        self.characters_list = selected_characters
        return movie_script,selected_characters

    def ScriptBreak(self, all_chat=[]):
        

        movie_script, characters_list = self.extract_characters_from_json(self.script_path, 40)
        first_sentence = movie_script.split(".")[0]
        # all_chat.append(query)
        previous_sub_script = None
        n = 0
        index = 1
        result = {}
        characters_list = str(characters_list)
        while True:
            if previous_sub_script: 
                query = f"""
                    Script Synopsis: {movie_script}
                    Character: {characters_list}
                    Previous Sub-Script: {previous_sub_script}
                    """
            else:
                query = f"""
                    Script Synopsis: {movie_script}
                    Character: {characters_list}
                    There is no Previous Sub-Script. 
                    The current sub-script is the first one. Please start summarizing the first sub-script based on the following content: {first_sentence}.
                    """
                
            query = f"""
                    Script Synopsis: {movie_script}
                    Character: {characters_list}
                    """
            
            task_response = self.screenwriter_agent(query, parse=True)
            # task_response = task_response.replace("'",'"')
            result = task_response

            break
        
        # all_chat.append(self.task_agent.messages)
        save_json(result, self.sub_script_path)
        # return 

    def ScenePlanning(self):
        data = self.read_json(self.sub_script_path)
        data_scene = data
        
        character_relationships = data['Relationships']
        sub_script_list = data['Sub-Script']

        for sub_script_name in sub_script_list:
            sub_script = sub_script_list[sub_script_name]["Plot"]
            query = f"""
                        Given the following inputs:
                        - Script Synopsis: "{sub_script}"
                        - Character Relationships: {character_relationships}
                        """
            task_response = self.sceneplanning_agent(query, parse=True)
            # if "Scene Annotation" not in data_scene[sub_script_name]:
            #     data_scene[sub_script_name]["Scene Annotation"] = []
            
            data_scene['Sub-Script'][sub_script_name]["Scene Annotation"]=task_response

            save_json(data_scene, self.scene_path)
            # break
    
    def ShotPlotCreate(self):
        data = self.read_json(self.scene_path)
        data_scene = data
        
        character_relationships = data['Relationships']
        sub_script_list = data['Sub-Script']

        for sub_script_name in sub_script_list:
            scene_list = sub_script_list[sub_script_name]["Scene Annotation"]["Scene"]
            for scene_name in scene_list:
                scene_details = scene_list[scene_name]
                query = f"""
                            Given the following Scene Details:
                            - Involving Characters: "{scene_details['Involving Characters']}" 
                            - Plot: "{scene_details['Plot']}"
                            - Scene Description: "{scene_details['Scene Description']}"
                            - Emotional Tone: "{scene_details['Emotional Tone']}"
                            - Key Props: {scene_details['Key Props']}
                            - Cinematography Notes: "{scene_details['Cinematography Notes']}"
                            """
                            
                task_response = self.shotplotcreate_agent(query, parse=True)
                # if "Shot Annotation" not in data_scene[sub_script_name]:
                #     data_scene[sub_script_name]["Shot Annotation"] = []
                
                data_scene['Sub-Script'][sub_script_name]["Scene Annotation"]["Scene"][scene_name]["Shot Annotation"] = task_response

                save_json(data_scene, self.shot_path)
            #     break
        
    def VideoAudioGen(self):
        data = self.read_json(self.shot_path)
        character_relationships = data['Relationships']
        sub_script_list = data['Sub-Script']

        for idx_1,sub_script_name in enumerate(sub_script_list):
            scene_list = sub_script_list[sub_script_name]["Scene Annotation"]["Scene"]
            # scene_path = os.path.join(self.video_save_path,shot_name+".jpg")
            # if idx_1!=len(sub_script_list)-1:
            #     continue

            for scene_name in scene_list:
                shot_lists = scene_list[scene_name]["Shot Annotation"]["Shot"]

                # scene_path = os.path.join(self.video_save_path,shot_name+".jpg")
                # if idx_1!=len(sub_script_list)-1:
                #     continue

                for shot_name in shot_lists:
                    shot_info = shot_lists[shot_name]
                    if self.sample_model == "ROICtrl":
                        plot = shot_info["Coarse Plot"]
                    else:
                        plot = shot_info["Plot/Visual Description"]

                    character_list = shot_info["Involving Characters"]
                    subtitle = shot_info["Subtitles"]
                    
                    character_phot_list = [os.path.join(self.character_photo_path,i.replace(" ","_"),"best.png") for i in character_list]
                    # save_path = os.path.join(self.video_save_path,shot_name+".jpg")
                    save_path = os.path.join(self.video_save_path,sub_script_name + "|" + scene_name + "|" + shot_name+".jpg")
                    save_path = save_path.replace(" ","_")

                    # if os.path.exists(save_path):
                    #     continue

                    print("Save the video to path:",save_path)
                    character_box = character_list

                    self.tools.sample(plot,character_phot_list,character_box,subtitle,save_path,(1024, 512))


                    for i,name in enumerate(subtitle):
                        wave_path = save_path.replace(".jpg","")+"_"+str(i)+"_"+name+".wav"
                        image_path = save_path

                        text_prompt = subtitle[name]

                    # break
                # break
            # break

    def Final(self):
        directory = self.video_save_path
        mp4_files = [f for f in os.listdir(directory) if f.endswith('.mp4')]

        mp4_files.sort()

        clips = []
        for file in mp4_files:
            file_path = os.path.join(directory, file)
            clip = VideoFileClip(file_path)
            clips.append(clip)

        final_video = concatenate_videoclips(clips)

        final_video_path = os.path.join(directory, "final_video.mp4")
        final_video.write_videofile(final_video_path, codec="libx264")
            
def main():
    args = parse_args()
    script_path = args.script_path
    character_photo_path = args.character_photo_path
    movie_director = ScriptBreakAgent(args,sample_model=args.gen_model, audio_model=args.audio_model, \
                                      talk_model=args.talk_model, Image2Video=args.Image2Video, script_path = script_path, \
                                    character_photo_path=character_photo_path, \
                                    save_mode="video")

    movie_director.ScriptBreak()
    movie_director.ScenePlanning()
    movie_director.ShotPlotCreate()
    movie_director.VideoAudioGen()
    # movie_director.AudioGen(script_path)
    movie_director.Final()


if __name__ == "__main__":
    main()











