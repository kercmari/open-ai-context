import json

class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    RESET = '\033[0m'

def clear_json(response_text):
    response_text_corrected = response_text.replace("'", '"')
    try:
        json_list= []
        if not response_text_corrected.endswith('"}]'):
            json_list =  (fix_truncated_json(response_text_corrected))
            json_list= json.loads(json_list)
        else :    
            json_list = json.loads(response_text_corrected)
        return json_list
    except json.JSONDecodeError as e:
        print("Error al decodificar la respuesta JSON:", str(e))
       
        return []
def fix_truncated_json(json_str):
    # Eliminar espacios en blanco al principio y al final de la cadena
    json_str = json_str.strip()
    
    # Añadir un cierre adecuado al JSON truncado
    if not json_str.endswith('"}]'):
        # Buscar la última coma
        last_text = json_str
        last_comma_index = json_str.rfind(',')
        if last_comma_index != -1:
            last_text = json_str[:last_comma_index]
        if not last_text.endswith('"}'):       
            # Reemplazar la última coma por un cierre de objeto y lista
            json_str = fix_truncated_json(last_text)
        else:
            json_str = json_str[:last_comma_index] + ']'
            
    
    return json_str
def empy_open_close(texto):
    aperturas_vacias = []
    cierres_vacios = []
    stack = []

    for i, char in enumerate(texto):
        if char in '{[':
            stack.append((char, i))
        elif char in '}]':
            if stack:
                apertura, apertura_index = stack.pop()
                if apertura == '{' and char == '}':
                    if apertura_index + 1 == i:
                        cierres_vacios.append((apertura_index, i))
                elif apertura == '[' and char == ']':
                    if apertura_index + 1 == i:
                        aperturas_vacias.append((apertura_index, i))
            else:
                pass
    
    return aperturas_vacias, cierres_vacios
