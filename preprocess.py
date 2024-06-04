# %%
import os
from aicsimageio import AICSImage
import pandas as pd
import numpy as np

# %%
def preprocess(lif_file):
    """
    Performs preprocessing for a given lif file.

    This function checks if the necessary directories and csv file exists, 
    and if they don't, it creates them. It also populates the csv file 
    with the necessary scene information.

    Args:
    lif_file (str): Path to the lif file.

    Returns:
    tuple: Paths to the processed directory, input directory, output directory,
    plots directory, images directory, and csv file.
    """
    if not os.path.isfile(lif_file):
        raise ValueError(f"{lif_file} does not exist or is not a valid file.")

    lif_dir = os.path.dirname(lif_file)
    filename = os.path.splitext(os.path.basename(lif_file))[0]

    # Define directories and csv file
    processed_dir = os.path.join(lif_dir, f"{filename}_processed")
    input_dir = os.path.join(processed_dir, 'input')
    output_dir = os.path.join(processed_dir, 'output')
    plots_dir = os.path.join(output_dir, 'plots')
    images_dir = os.path.join(output_dir, 'images')
    condition_info_file = os.path.join(input_dir, 'condition_info.csv')
    ch_info_file = os.path.join(input_dir, 'ch_info.csv')

    directories = [processed_dir, input_dir, output_dir, plots_dir, images_dir]

    # Create directories if they don't exist
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    lif_stack = AICSImage(lif_file)

    # Create csv file if it doesn't exist
    if not os.path.exists(condition_info_file):
        print('condition_info.csv does not exist. Please fill it.')
        scenes = [scene.split('/')[-1] for scene in lif_stack.scenes]
        df = pd.DataFrame(scenes, columns=['scene']) # Convert scenes to dataframe
        df['condition'] = '' # Add condition column with empty value
        df.to_csv(condition_info_file, index=False) # Save dataframe in input_dir as csv

    if not os.path.exists(ch_info_file):
        print('ch_info.csv does not exist. Please fill it.')
        df_ch = pd.DataFrame(lif_stack.channel_names, columns=['color']) # ch_list to pd.dataframe
        df_ch.to_csv(ch_info_file, index=True) # save as csv

    return processed_dir, input_dir, output_dir, plots_dir, images_dir, condition_info_file, ch_info_file

def preprocess_txt(txt_file):
    """
    Performs preprocessing for a given txt file.

    This function checks if the necessary directories and csv file exists, 
    and if they don't, it creates them. It also populates the csv file 
    with the necessary scene information.

    Args:
    txt_file (str): Path to the txt file.

    Returns:
    tuple: Paths to the processed directory, input directory, output directory,
    plots directory, images directory, and csv file.
    """
    if not os.path.isfile(txt_file):
        raise ValueError(f"{txt_file} does not exist or is not a valid file.")

    txt_dir = os.path.dirname(txt_file)
    filename = os.path.splitext(os.path.basename(txt_file))[0]

    # Define directories and csv file
    processed_dir = os.path.join(txt_dir, f"{filename}_processed")
    input_dir = os.path.join(processed_dir, 'input')
    output_dir = os.path.join(processed_dir, 'output')
    plots_dir = os.path.join(output_dir, 'plots')
    condition_info_file = os.path.join(input_dir, 'condition_info.csv')

    directories = [processed_dir, input_dir, output_dir, plots_dir]

    # Create directories if they don't exist
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    df = pd.read_csv(txt_file, sep='\t', skiprows=2, index_col=1, encoding='utf-16')
    df = df.dropna(axis=1, how='all')
    df = df.reset_index(drop=True)
    df = df.dropna()
    df = df.replace('#SAT', np.nan)
    df.index = pd.to_timedelta(df['Time']).dt.total_seconds() / 60
    df.index.name = 'Time (min)'
    df = df.drop(columns=['Time'])
    # Create csv file if it doesn't exist
    if not os.path.exists(condition_info_file):
        print('condition_info.csv does not exist. Please fill it.')
        df_cond = pd.DataFrame(df.columns, columns=['well'])
        df_cond['Condition'] = ""
        df_cond.to_csv(condition_info_file, index=False)

    return processed_dir, input_dir, output_dir, plots_dir, condition_info_file, df

def preprocess_cfxfile(raw_file, dt, sheet_name=0):
    """
    Performs preprocessing for a given cfx file.

    This function checks if the necessary directories and csv file exists, 
    and if they don't, it creates them. It also populates the csv file 
    with the necessary scene information.

    Args:
    raw_file (str): Path to the cfx file.

    Returns:
    tuple: Paths to the processed directory, input directory, output directory,
    plots directory, images directory, and csv file.
    """
    if not os.path.isfile(raw_file):
        raise ValueError(f"{raw_file} does not exist or is not a valid file.")

    raw_dir = os.path.dirname(raw_file)
    filename = os.path.splitext(os.path.basename(raw_file))[0]

    # Define directories and csv file
    processed_dir = os.path.join(raw_dir, f"{filename}_processed")
    input_dir = os.path.join(processed_dir, 'input')
    output_dir = os.path.join(processed_dir, 'output')
    plots_dir = os.path.join(output_dir, 'plots')
    condition_info_file = os.path.join(input_dir, 'condition_info.csv')

    directories = [processed_dir, input_dir, output_dir, plots_dir]

    # Create directories if they don't exist
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    df = pd.read_excel(raw_file, sheet_name=sheet_name, index_col='Cycle', header=0)
    # if column name is 'Unnamed: 0', drop it
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns='Unnamed: 0')
    df = df.apply(pd.to_numeric, errors='coerce')
    df.index = (df.index * dt) - dt
    df.index.name = 'Time (min)'
    # Create csv file if it doesn't exist
    if not os.path.exists(condition_info_file):
        print('condition_info.csv does not exist. Please fill it.')
        df_cond = pd.DataFrame(df.columns, columns=['well'])
        df_cond['Condition'] = ""
        df_cond.to_csv(condition_info_file, index=False)

    return processed_dir, input_dir, output_dir, plots_dir, condition_info_file, df