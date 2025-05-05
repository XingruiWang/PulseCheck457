import os
import json

def remove_material_and_filter_questions(data):
    """Removes <M>, <M2>, <M3>, their corresponding params and side_inputs, and filters out questions with Material."""
    filtered_data = []
    
    for item in data:
        # Check if the item contains any 'Material' in params
        # if any(param.get("type") == "Material" for param in item.get("params", [])):
        #     continue  # Skip the item if it has 'Material' in params
        
        # Modify the 'text' field by removing <M>, <M2>, and <M3>
        modified_texts = [text.replace("<M> ", "").replace("<M2> ", "").replace("<M3> ", "").replace("<M4> ", "") for text in item.get("text", [])]
        have_material = [text for text in modified_texts if "material" in text]
        if have_material:
            continue
        
        # Modify the 'params' field by removing 'Material' type params
        modified_params = [param for param in item.get("params", []) if param.get("type") != "Material"]
        
        # Modify the 'side_inputs' field by removing <M>, <M2>, and <M3>
        for node in item.get("nodes", []):
            if "side_inputs" in node:
                node["side_inputs"] = [si for si in node["side_inputs"] if si not in ("<M>", "<M2>", "<M3>", "<M4>")]
        
        # Update the item with the modified text, params, and nodes
        item['text'] = modified_texts
        item['params'] = modified_params
        
        # Check if any params contain an item with type "Relation" (i.e., <R>)
        if any(param.get("type") == "Relation" for param in item.get("params", [])):
            filtered_data.append(item)
    
    return filtered_data

def process_files(input_directory, output_directory, output_suffix='_filtered.json'):
    """Processes all .json files in the input directory, modifies and saves filtered files to the output directory."""
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        
    for file_name in os.listdir(input_directory):
        if file_name.endswith('.json'):
            # Construct the full file path
            file_path = os.path.join(input_directory, file_name)
            
            # Open and read the content of the file
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Modify and filter the data
            modified_filtered_data = remove_material_and_filter_questions(data)
            
            # Define the output file path with a new name in the output directory
            output_file_path = os.path.join(output_directory, file_name.replace('.json', output_suffix))
            
            # Save the modified and filtered questions into a new file
            with open(output_file_path, 'w', encoding='utf-8') as f_out:
                json.dump(modified_filtered_data, f_out, indent=4)
            
            print(f"Processed and saved: {output_file_path}")

# Define the input directory (where your original .json files are located)
input_directory = '/home/xingrui/publish/superclevr_3D_questions/question_generation/CLEVR_1.0_templates'

# Define the output directory (where the filtered files will be saved)
output_directory = '/home/xingrui/publish/superclevr_3D_questions/question_generation/super_6d_questions'

# Process all JSON files in the input directory and save to the output directory
process_files(input_directory, output_directory)
