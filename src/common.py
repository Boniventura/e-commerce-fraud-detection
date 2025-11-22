import os 
import matplotlib.pyplot as plt

def save_plot(folder_path = None, file_name = None):
    try: 
        full_path = f'{folder_path}/{file_name}'

        os.makedirs(folder_path, exist_ok=True) 
        
        plt.savefig(full_path, dpi=300, bbox_inches='tight')

        print(f"Wykres zapisano w: {full_path}")

    except Exception as e:
        print(f"Nie udało się zapisać wykresu. Błąd: {e}")