import os

def replace_string_in_files(directory, target_str, replacement_str="<TOK>"):
    # Loop through all files in the directory
    for filename in os.listdir(directory):
        # Check if the file is a .txt file
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            
            # Read the file content
            with open(file_path, 'r', encoding='utf-8') as file:
                file_content = file.read()
            
            # Replace the target string with the replacement
            new_content = file_content.replace(target_str, replacement_str)

            if "<TOK>" not in new_content:
                new_content = file_content.replace(target_str.split(" ")[0], replacement_str)
                try:
                    new_content = file_content.replace(target_str.split(" ")[1], "")
                except:
                    pass
                if "<TOK>" not in new_content:
                    new_content = "<TOK>"
            
            # Write the updated content back to the file
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(new_content)


# Directory containing the .txt files
directory_path = '/storage/wuweijia/MovieGen/lsmdc/GT/Character_Bank/Char_Desc_AutoStory/Test/1041_This_is_40'

for character in os.listdir(directory_path):
    target_string = character.split("-")[-1].replace("_"," ")

    print(target_string)
    # String you want to replace
    # target_string = 'your_string_to_replace'

    # Call the function
    replace_string_in_files(os.path.join(directory_path,character), target_string)
