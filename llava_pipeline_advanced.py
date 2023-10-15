#%% - Imports
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

import numpy as np
import pandas as pd
from PIL import Image as img
from datetime import datetime

#%% - Prepare LLaVA
model_path = "liuhaotian/llava-llama-2-13b-chat-lightning-preview"
model_name = get_model_name_from_path(model_path)
model_base = None

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=model_base,
    model_name=model_name
)

model = model.to('cpu')

#%% - Define run model function
def run_llava(file_path, prompt):
    model_path = "liuhaotian/llava-llama-2-13b-chat-lightning-preview"
    model_name = get_model_name_from_path(model_path)
    model_base = None
    model_prompt = prompt
    image_file = file_path

    args = type('Args', (), {
        "model_path": model_path,
        "model_base": model_base,
        "model_name": model_name,
        "query": model_prompt,
        "conv_mode": None,
        "image_file": image_file
    })()

    model_response = eval_model(args)
    return model_response

#%% - Helper functions
def df_to_excel(df: pd.DataFrame, file_name: str):
    now = datetime.now()
    short_format_date_time = now.strftime("%Y-%m-%d_%H-%M-%S")
    
    path = r"Z:\Research\003_Max Konrad\01_Multimodal Marketing\01_Data\model_output" + "\\" + file_name + "_" + short_format_date_time + ".xlsx"
    with pd.ExcelWriter(path) as writer:
        df.to_excel(writer, index=False)

def run_inference(df):
    total_rows = len(df)
    
    # Ensure the 'output_LLaVA' column exists and has the correct data type (string)
    df['output_LLaVA'] = df.get('output_LLaVA', pd.Series(dtype='str'))
    
    # Iterating through all rows in the dataframe
    for index, row in df.iterrows():
        # Print progress
        print(f"Processing row {index+1} of {total_rows}...")
        
        # Saving values from columns "file_path" and "prompt" into variables
        file_path = row['file_path']
        prompt = row['prompt']

        print(f"Task: {prompt}")

        try:
            inference_output = run_llava(file_path, prompt)
        except Exception as e:
            print(f"An error occurred: {e}")
            inference_output = "n/a"
        df.at[index, 'output_LLaVA'] = inference_output
    
    # Saving the modified dataframe to an Excel file using your custom function
    df_to_excel(df, "output")

#%% - Main function
# Import XLS with prompts and run inference
df1 = pd.read_excel(r"Z:\Research\003_Max Konrad\01_Multimodal Marketing\01_Data\model_output\Backup_Luminous\3of3_AVA_2023-09-19_11-19-18.xlsx", sheet_name=0)
run_inference(df1)
# %%
