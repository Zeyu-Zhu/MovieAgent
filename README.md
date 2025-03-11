# MovieAgent




<table align="center">
  <tr>
    <td><img src="./assets/logo.png" alt="MovieAgent Logo" width="180"></td>
    <td>
      <h3>MovieAgent: Automated Movie Generation via Multi-Agent CoT Planning</h3>
      <a href="https://weijiawu.github.io/MovieAgent/">
        <img src="https://img.shields.io/static/v1?label=Project%20Page&message=Github&color=blue&logo=github-pages">
      </a> &ensp;
      <a href="https://arxiv.org/abs/2503.07314">
        <img src="https://img.shields.io/static/v1?label=Paper&message=Arxiv&color=red&logo=arxiv">
      </a>
    </td>
  </tr>
</table>



![MovieAgent Demo](./assets/demo.gif)



## :notes: **Updates**

<!--- [ ] Mar. 13, 2024. Release the train code in **three month**.-->

- [ ] Mar. 18, 2024. Release the inference code.
- [x] Mar. 10, 2025. Rep initialization (No code).


## 🐱 Abstract
Existing long-form video generation frameworks lack automated planning and often rely on manual intervention for storyline development, scene composition, cinematography design, and character interaction coordination, leading to high production costs and inefficiencies. To address these challenges, we present MovieAgent, an automated movie generation via multi-agent Chain of Thought (CoT) planning. MovieAgent offers two key advantages: 1) We firstly explore and define the paradigm of automated movie/long-video generation. Given a script and character bank, our MovieAgent can generates multi-scene, multi-shot long-form videos with a coherent narrative, while ensuring character consistency, synchronized subtitles, and stable audio throughout the film. 2) MovieAgent introduces a hierarchical CoT-based reasoning process to automatically structure scenes, camera settings, and cinematography, significantly reducing human effort. By employing multiple LLM agents to simulate the roles of a director, screenwriter, storyboard artist, and location manager, MovieAgent streamlines the production pipeline. Our framework represents a significant step toward fully automated movie production, bridging the gap between AI-driven video generation and high-quality, narrative-consistent filmmaking.

---

<p align="center">
<img src="./assets/structure.png" width="800px"/>  
<br>
</p>



<a name="installation"></a>
## :hammer: Installation

1. Clone the repository.

```bash
git clone 
cd MovieAgent
```

2. Install the environment.
```bash
conda create -n MovieAgent python=3.8
conda activate MovieAgent
pip install -r requirements.txt
```



<a name="usage"></a>
## Model and Data Preparation





First, you need to prepare the `script synopsis of movie` and  `photo,audio of character` as follow:


```
dataset/
    movie name/
        script_synopsis.json
        character_list/
            character 1/
                photo_1.jpg
                photo_2.jpg
                audio.wav
            character 2/
            ...
```


Then, you need to configure the `open_api_key` and `model name` in `movie_agent/script/run.sh`.

The reasoning process using agents may involve various image and video generation models. Most models can be automatically downloaded, while a few require **manual configuration**, such as StoryDiffusion and ROICtrl for **character customization**:

### Supported Model Zoo  

| LLM (Language Model) | Image Gen. (Image Generation) | Video Gen. (Video Generation) |
|----------------------|------------------------------|------------------------------|
| GPT4-o               | ROICtrl                   | SVD                            |



### Image Gen. Model - ROICtrl 
You can download the our weight from [Google drive](www.google.com) directly or train by yourself following the steps:

```bash
cd tools/Mix-of-Show
```
Step 1: generate the description for each character
```bash
bash Script/Step_1_generate_desc_char.sh
```

Step 2: Generate the data .json file and .json file  for each Character to train the lora
```bash
bash Script/Step_2_generate_data_json_for_character.sh
```

Step 3: train lora for each character
```bash
export PYTHONPATH=/storage/wuweijia/MovieGen/MovieDirector/MovieDirector/tools/Mix-of-Show:$PYTHONPATH 

for file in /storage/wuweijia/MovieGen/MovieDirector/MovieDirector/tools/Mix-of-Show/options/train/EDLoRA/MovieGen/InsideOut2/*.yml; do
    CUDA_VISIBLE_DEVICES=2 accelerate launch train_edlora.py -opt "$file"
done
```


Step 4: Generate config .json file for multi lora
```bash
bash Script/Step_4_generate_config_json_for_multi_lora.sh
```

Step 5: train multi lora 

```bash
bash train_lora_multi.sh
```

Step 6: Test

```bash
CUDA_VISIBLE_DEVICES=4  python sample_single_for_fusion.py
```




### Generate Movie/Long Video

Gnerate the long video with MovieDirector:
```bash
cd movie_agent
sh script/run.sh
```


## Citation

If you find our repo useful for your research, please consider citing our paper:

    @misc{wu2025movieagent,
          title={Automated Movie Generation via Multi-Agent CoT Planning}, 
          author={Weijia Wu, Zeyu Zhu, Mike Zheng Shou},
          year={2025},
          eprint={2503.07314},
          archivePrefix={arXiv},
          primaryClass={cs.CV}
    }

## 🤗Acknowledgements
- Thanks to [Diffusers](https://github.com/huggingface/diffusers) for the wonderful work and codebase.
- Thanks to [Evaluation-Agent](https://github.com/Vchitect/Evaluation-Agent) for the wonderful work and codebase.


