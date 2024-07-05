## Documentación de Web Scraping Inteligente con IA

### ENFOQUE

El objetivo de este proyecto es desarrollar un sistema de web scraping inteligente que utilice inteligencia artificial para clasificar el contenido extraído de sitios web. Utilizaremos modelos de OpenAI para la clasificación y comparación con otros modelos disponibles. El cual hemos analizada que se debe realizar un  preanálisis de la información que implica tokenización, normalizacion, lematización, eliminación de stopwords y análisis de dependencias. Estos pasos preparan el texto para análisis avanzados. spaCy, gyms o  gensim ofrece un conjunto completo de herramientas para el preanálisis,mientras que Transformers proporciona modelos preentrenados para interpretar preguntas y clasificar información con gran precisión. 

Se han realizado pruebas con modelos open source como BERT, T5, ROBERTA o DISTILBERT, para el analisis de la informacion de acuerdo a una soliciuto o pregunta generica ha dado resultados enfocados a filtros o similitud de coindicencias enviadas al modelo. El cual no ha dado respuestas esperadas, es decir  usarlos como modelo para segementacion de data deacuerdo a un requerimiento con una buena probailidad interprestacion, se requiere de IA Generativa.
### Pricing

![Alt text](image.png)
### Tecnologías Utilizadas

- **OpenAI GPT-4**: Para clasificación y análisis de contenido.

- **Requests**: Para realizar solicitudes HTTP.
- **spaCy**: Para procesamiento de lenguaje natural.
- **GYMS**: Para tokenización y normalización.
- **Gensim**: Para preprocesamiento de texto.
- **Transformers**: Para trabajar con modelos de lenguaje.

### Beneficios de Usar OpenAI

- **Precisión**: Los modelos de OpenAI, como GPT-4, son conocidos por su alta precisión en tareas de procesamiento de lenguaje natural.
- **Versatilidad**: Capacidad de manejar diversas tareas de lenguaje sin necesidad de ajustes específicos.
- **Facilidad de Integración**: API fácil de usar que permite una integración sencilla con sistemas existentes.
- **Actualizaciones Continuas**: OpenAI actualiza regularmente sus modelos para mejorar su rendimiento y capacidades.
- Extraccion de informacion mas completa para contenidos complejos



### Limitaciones de OpenAI

- **Costo**: El uso de la API de OpenAI puede ser costoso, especialmente para grandes volúmenes de datos.
- **Latencia**: Dependiendo de la complejidad de la tarea, el tiempo de respuesta puede ser más alto en comparación con otros modelos locales.
- **Privacidad**: Los datos enviados a la API deben ser manejados con cuidado para cumplir con regulaciones de privacidad.

### Comparación con Otros Modelos

| Característica       | OpenAI GPT-4 | BERT             | RoBERTa          | DistilBERT       |
| -------------------- | ------------ | ---------------- | ---------------- | ---------------- |
| **Precisión**        | Alta         | Alta             | Muy Alta         | Media            |
| **Versatilidad**     | Alta         | Media            | Media            | Media            |
| **Facilidad de Uso** | Muy Alta     | Media            | Media            | Alta             |
| **Costo**            | Alto         | Bajo             | Medio            | Bajo             |
| **Latencia**         | Media        | Baja             | Baja             | Muy Baja         |
| **Actualizaciones**  | Frecuentes   | Menos Frecuentes | Menos Frecuentes | Menos Frecuentes |
| **Privacidad**       | Baja (API)   | Alta (Local)     | Alta (Local)     | Alta (Local)     |

### Pasos para Implementar el Sistema

1. **Configuración del Entorno**
    Crear el ambiente
    ```
    python -m venv env
    env\Scripts\activate
    source env/bin/activate

    pip install -r requirements.txt
    python -m spacy download en_core_web_sm

    ```

    Instalar las librerías necesarias:   
        `beautifulsoup4`, `requests`, `spacy`, `gyms`, `gensim`, `transformers`, `openai`.

        
        ```bash
        pip install beautifulsoup4 requests spacy gyms gensim transformers openai
        ```
2. **Config JSON**
- `"path_new"`: Ruta al archivo de datos a procesar.
- `"pregunta"`: Pregunta o instrucción para el sistema de procesamiento, la solicitud que se le va a enviar al API de IA.
- `"rules"` (reglas):
    Se debe establecer todas las reglas para el prompt del requerimiento
  - "Be concise": Sé conciso en las respuestas generadas.
  - "Maximum 270 tokens or 1100 response characters": Límite máximo de tokens o caracteres en la respuesta.
  - "You are a wizard that generates lists of JSON objects. Provides a continuous list of JSON objects within a single set of square brackets. Make sure the listing and each JSON object are properly closed!": Descripción de cómo debe ser la estructura de las respuestas generadas.
  - "You are an expert at extracting and interpreting information based on all the context": Descripción del papel del sistema en el procesamiento de la información.
- `"formats"` (formatos):
    Se establece el tipo de formato a recibir los datps
  - "El objeto debe tener el formato [{'keyOne': value, 'keyTow': value.. },{'keyOne': value, 'keyTow': value.. }]": Formato esperado de los objetos JSON generados.
- `"max_tokens"`: Límite máximo de tokens para la generación de texto, este texto servira como INPUT Token para el servicio de IA, para OPEN AI token de entrada/salida en GPT3 es de hasta 2048 tokens , para GPT4 es de 4096 tokens.
- `"max_response_tokens"`: Límite máximo de tokens en la respuesta generada por parte del API OpenAI.
- `"n"`: Número de respuestas a generar por parte de la IA, un solo input de entrada puede tener multiples respuesta.
- `"temperature"`: Parámetro de temperatura para el modelo de generación de texto, est controla la aleatoriedad y creatividad de la generación de texto.
- `"limit"` (opcional): Límite para cuantos request a IA se quiere enviar por cada JSON DATA escrapeada, caso contrario puede ir nullo, es decir toma todos los datos y los grupos de tokens.
- `"indexRead"` (opcional):Desde que elemento se desea hacer la lectura de la JSON DATA escrapeada .
- `"typeFormat"`: Tipo de formato esperado para la respuesta generada (en este caso, "JSON").

3. **Codigo con Open AI**
    ```python
    import json
import time
import openai
import os
import pandas as pd
import spacy
from gensim.utils import simple_preprocess
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Iniciar sesión en Hugging Face si es necesario
# login("tu-token-de-hugging-face")

# Identificador y carga del modelo BERT desde Hugging Face
model_id = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

# Cargar el modelo de spaCy
nlp = spacy.load("en_core_web_sm")

# Función para procesar el texto con spaCy
# Esta función utiliza spaCy para lematizar el texto y eliminar las palabras vacías
def process_text(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop]
    return " ".join(tokens)

# Función para tokenizar y normalizar el texto con Gensim
# Esta función utiliza Gensim para tokenizar el texto y eliminar acentos
def gensim_preprocess(text):
    tokens = simple_preprocess(text, deacc=True)  # deacc=True elimina acentos
    return tokens

# Función para leer archivo
# Esta función lee un archivo CSV, JSON o TXT y lo convierte en un DataFrame de pandas
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

# Función para detectar aperturas y cierres vacíos en un texto
# Esta función detecta y devuelve los índices de corchetes y llaves vacíos
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
                # Manejo de caso de cierre sin apertura correspondiente
                pass
    
    return aperturas_vacias, cierres_vacios

# Función para dividir el texto en fragmentos
# Esta función divide el texto en fragmentos de una longitud máxima especificada
def split_into_fragments(text, max_length=500):
    tokens = tokenizer(text)["input_ids"]
    return [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]

# Función para limpiar y decodificar una respuesta JSON
# Esta función corrige el formato JSON y lo decodifica
def clear_json(response_text):
    response_text_corrected = response_text.replace("'", '"')
    try:
        json_list = json.loads(response_text_corrected)
        return json_list
    except json.JSONDecodeError as e:
        print("Error al decodificar la respuesta JSON:", str(e))
        return []

# Función para agrupar fragmentos en base al límite de tokens
# Esta función agrupa fragmentos de texto en base a un límite máximo de tokens
def group_fragments(fragments, max_tokens):
    grouped_fragments = []
    current_group = []
    current_length = 0

    for fragment in fragments:
        fragment_length = len(fragment)
        if current_length + fragment_length > max_tokens:
            grouped_fragments.append(" ".join(current_group))
            current_group = [fragment]
            current_length = fragment_length
        else:
            current_group.append(fragment)
            current_length += fragment_length

    if current_group:
        grouped_fragments.append(" ".join(current_group))

    return grouped_fragments

# Función para clasificar texto con OpenAI
# Esta función utiliza la API de OpenAI para clasificar texto basado en una pregunta y un contexto dado
def filter_with_gpt3(texts, question, role, model_name='gpt-3.5-turbo', max_tokens=150, n=1, temperature=0.7):
    openai.api_key = ''  # Proporcione su API key de OpenAI
    
    context = ' '.join(texts)
    prompt = f"Contexto: {context}\nPregunta: {question}\nRespuesta:"

    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "system", "content": role},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        n=n,
        stop=None,
        temperature=temperature,
    )

    answers = [choice['message']['content'].strip() for choice in response['choices']]
    return answers

# Función para construir el rol de "system"
# Esta función construye el contenido del rol "system" para la API de OpenAI basado en reglas y formatos especificados
def construct_system_role(rules, formats):
    role_content = "Eres un asistente experto en responder preguntas basadas en contexto y de forma concisa. "
    if rules:
        role_content += "Reglas: " + ", ".join(rules) + ". "
    if formats:
        role_content += "Formatos de respuesta: " + ", ".join(formats) + "."
    return role_content

# Función principal para ejecutar el flujo completo
# Esta función coordina la lectura de datos, procesamiento de texto, y clasificación utilizando la API de OpenAI
def main(path_new, pregunta, rules, formats, max_tokens=3048, max_response_tokens=150, n=1, temperature=0.7, limit=None, indexRead=None, typeFormat='JSON'):
 # La clase Colors define constantes para colorear el texto en la consola, lo que facilita la visualización de mensajes de estado y errores.

    class Colors:
        RED = '\033[91m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        BLUE = '\033[94m'
        MAGENTA = '\033[95m'
        CYAN = '\033[96m'
        RESET = '\033[0m'
    
# leer_archivo(path_new) carga el archivo de datos especificado por path_new y devuelve un DataFrame.


    df = leer_archivo(path_new)

# Procesar y tokenizar el contenido
# Procesamiento del DataFrame:
# El bucle for _, row in df.iterrows() itera sobre cada fila del DataFrame.
# Para cada fila, se itera sobre cada clave y valor.
# Si el valor es una lista, se procesa cada elemento de la lista.
# Si el valor es una cadena, se procesa directamente.
# process_text(text) y gensim_preprocess(processed_text) se utilizan para limpiar y tokenizar el texto.
# Los textos procesados se almacenan en processed_texts.
    processed_texts = []
    for _, row in df.iterrows():
        for key, value in row.items():
            if isinstance(value, list):
                for text in value:
                    processed_text = process_text(text)
                    gensim_processed_tokens = gensim_preprocess(processed_text)
                    processed_texts.append(" ".join(gensim_processed_tokens))
            elif isinstance(value, str):
                processed_text = process_text(value)
                gensim_processed_tokens = gensim_preprocess(processed_text)
                processed_texts.append(" ".join(gensim_processed_tokens))

    # Crear la matriz de contextos
 # Se tokeniza cada texto procesado en fragmentos de longitud máxima 500.
# Cada fragmento se decodifica y se añade a context_matrix.

    context_matrix = []
    for text in processed_texts:
        tokenized_fragments = split_into_fragments(text, max_length=500)
        for fragment in tokenized_fragments:
            context_matrix.append(tokenizer.decode(fragment, skip_special_tokens=True))

# Agrupar fragmentos en base al límite de tokens    permitidos por OpenAI
# group_fragments(context_matrix, max_tokens) agrupa los fragmentos de texto en función del límite de tokens especificado por max_tokens.

    grouped_contexts = group_fragments(context_matrix, max_tokens)

# Construir el rol de "system"
# construct_system_role(rules, formats) construye un rol de sistema a partir de las reglas y formatos proporcionados.

    role = construct_system_role(rules, formats)

# Pregunta dada
# Se itera sobre cada grupo en gropu_new.
# Cada grupo se envía a la API de OpenAI con filter_with_gpt3.
# La respuesta se procesa y se formatea según sea necesario (lista o JSON).
# Los resultados se almacenan en results.

    results = []

    total = len(grouped_contexts)
    print('Total de requests', total)
    
    # Procesar cada grupo con OpenAI
    completo = len(grouped_contexts) + 1
    indexInit = 0
    index = completo
    if limit is not None:
        index = limit
    if indexRead is not None:
        indexInit = indexRead
        index = indexInit
    print(index, indexInit)
    gropu_new = grouped_contexts[indexInit:index]
    i = indexInit
    for group in gropu_new:
        try:
            print(Colors.BLUE, len(group), 'Tokens en el grupo', i, '/', total, Colors.RESET)
            inicio_tiempo = time.time()
            response = filter_with_gpt3(group, pregunta, role, max_tokens=max_response_tokens, n=n, temperature=temperature)
            response_elem = response[0]

            if isinstance(response, list) or typeFormat.lower() == "list":
                json_list = clear_json(response_elem)
                if len(json_list) > 1 and isinstance(json_list, list):
                    results.extend(json_list)
                else:
                    print(Colors.RED, 'Esta es la respuesta', len(response_elem), response_elem, Colors.RESET)
                    results.append(clear_json(response_elem))
            elif isinstance(response, dict) or typeFormat.lower() == "json":
                results.append(json.loads(response))
            else:
                results.append(json.loads(response))

            print(f"Tiempo de procesamiento: {time.time() - inicio_tiempo} segundos")
            time.sleep(10)
        except Exception as e:
            print(f"Error procesando el grupo {i}: {e}")
        i += 1

    print('Proceso de Guardado...')
# Guardar resultados en un archivo JSON sin sobrescribir datos anteriores
# Si el archivo de resultados ya existe, se cargan los datos existentes.
# Los nuevos resultados se añaden a los datos existentes.
# Los datos actualizados se guardan en output_file.

    output_file = 'output_results.json'
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
    else:
        existing_data = []

    existing_data.extend(results)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)

    print(f"Resultados guardados en {output_file}")

# Ejecución principal
# Esta función `main` está diseñada para procesar un archivo de datos, extraer información específica y generar resultados utilizando OpenAI. Los parámetros de configuración se especifican en un archivo JSON externo.


if __name__ == "__main__":
    path_new = "/home/kerly/data-complete/nlp/src/data2.json"
    pregunta = "Extrae me especialidades médicas, listado con dos key especializacion y descripcion, cada key con su respectivo valor"
    
    rules = ["Be concise", "Maximum 270 tokens or 1100 response characters", "You are a wizard that generates lists of JSON objects. Provides a continuous list of JSON objects within a single set of square brackets. Make sure the listing and each JSON object are properly closed!", 'You are an expert at extracting and interpreting information based on all the context']
    formats = ["El objeto debe tener el formato [{'keyOne': value, 'keyTow': value.. }, {'keyOne': value, 'keyTow': value.. }]"]
  
    max = 300
    limit = 2
    temperature = 0.5
    main(path_new, pregunta, rules, formats, temperature=temperature, limit=limit, max_response_tokens=max)

    ```