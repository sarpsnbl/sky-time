import json
import os
import shutil

def get_top_hardest_images(log_filepath, source_folder, dest_folder, x=10):
    image_errors = []
    
    # Read the jsonl file and calculate errors for each image
    with open(log_filepath, 'r') as file:
        for line in file:
            data = json.loads(line)
            
            if data.get('type') == 'image':
                pred_min = data['pred_min']
                actual_min = data['actual_min']
                
                # Calculate the absolute error
                error = abs(pred_min - actual_min)
                
                # Extract just the filename (e.g., 'DSCF2073.jpg' from 'dataset_512/DSCF2073.jpg')
                filename = os.path.basename(data['path'])
                
                image_errors.append((filename, error))
                
    # Sort the list by error in descending order (highest error first)
    image_errors.sort(key=lambda item: item[1], reverse=True)
    
    # Get the top x hardest images
    top_x_hardest = image_errors[:x]
    
    # Create the destination folder if it doesn't exist
    os.makedirs(dest_folder, exist_ok=True)
    
    print(f"Top {x} hardest images to predict:")
    
    # Copy the files
    for filename, error in top_x_hardest:
        source_path = os.path.join(source_folder, filename)
        dest_path = os.path.join(dest_folder, filename)
        
        if os.path.exists(source_path):
            shutil.copy2(source_path, dest_path)
            print(f"Copied: {filename} | Error: {error:.2f}")
        else:
            print(f"File not found: {source_path}")

# --- Example Usage ---
# Ensure your working directory contains the 'train_log.jsonl' and the 'dataset' folder
if __name__ == "__main__":
    LOG_FILE = "checkpoints/train_log.jsonl"
    SOURCE_DIR = "dataset_512"
    DEST_DIR = "hardest"
    TOP_X = 100
    
    get_top_hardest_images(LOG_FILE, SOURCE_DIR, DEST_DIR, TOP_X)