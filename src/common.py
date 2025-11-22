import os 
import matplotlib.pyplot as plt
import pandas as pd
import logging
logger = logging.getLogger(__name__)

def save_plot(folder_path = None, file_name = None):
    try: 
        full_path = f'{folder_path}/{file_name}'

        os.makedirs(folder_path, exist_ok=True) 

        plt.savefig(full_path, dpi=300, bbox_inches='tight')

        print(f"plot saved in: {full_path}")

    except Exception as e:
        print(f"Can not save the plot. Error: {e}")

