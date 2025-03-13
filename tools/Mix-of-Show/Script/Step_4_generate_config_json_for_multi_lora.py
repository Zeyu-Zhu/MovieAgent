import os
import json
import argparse



def parse_args():
    parser = argparse.ArgumentParser(description='MovieDirector', formatter_class=argparse.RawTextHelpFormatter)

    
    parser.add_argument(
        "--input_directory",
        type=str,
        required=True,
        help="user query",
    )
    parser.add_argument(
        "--output_directory",
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



def modify_and_save_json_files(input_dir, output_dir, movie_name):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    movie_list = os.listdir(input_dir)
    json_name = ""
    json_list = "["

    for index,name in enumerate(movie_list):
        
        character_name = name
        str_ = f"""{{
        "lora_path": "experiments/{movie_name}/{character_name}/models/edlora_model-latest.pth",
        "unet_alpha": 1.0,
        "text_encoder_alpha": 1.0,
        "concept_name": "<{character_name}1> <{character_name}2>"
    }}
"""     
        if index == 0:
            json_name = name
        else:
            json_name = json_name+"+"+name

        if index == len(movie_list)-1:
            json_list+=str_
        else:
            json_list+=str_
            json_list+=","

        

    json_list+="]"
    output_file_path = os.path.join(output_dir,json_name+".json")
    # Write the modified content to the new file
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.write(json_list)


def main():
    args = parse_args()
    # Call the function to create the JSON
    modify_and_save_json_files(args.input_directory, args.output_directory, args.movie_name)




if __name__ == "__main__":
    main()

