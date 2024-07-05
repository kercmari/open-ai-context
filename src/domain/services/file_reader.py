import os
import json
import pandas as pd

def leer_archivo(nombre_archivo):
    
    _, extension = os.path.splitext(nombre_archivo.lower())
    if extension == '.csv':
        data_df = pd.read_csv(nombre_archivo)
    elif extension == '.json':
        with open(nombre_archivo, 'r', encoding='utf-8') as f:
            data = json.load(f)
        data_df = pd.DataFrame(data)
    elif extension == '.txt':
        data_df = pd.read_csv(nombre_archivo, delimiter=',')
    else:
        raise ValueError(f"Extensión de archivo no compatible: {extension}")
    return data_df
