import datetime
import os
import json
import time
from ..domain.services.file_reader import leer_archivo
from ..domain.services.text_processor import process_text, gensim_preprocess, split_into_fragments, group_fragments
from ..domain.services.openai_service import filter_with_gpt3, construct_system_role
from ..domain.services.utils import clear_json, Colors
from ..domain.models.nlp_models import nlp, tokenizer

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as config_file:
        return json.load(config_file)

def process_dataframe(df):
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
    return processed_texts
def build_context_matrix(processed_texts, max_length=500):
    context_matrix = []
    for text in processed_texts:
        tokenized_fragments = split_into_fragments(text, max_length)
        for fragment in tokenized_fragments:
            context_matrix.append(tokenizer.decode(fragment, skip_special_tokens=True))
    return context_matrix


def process_groups(grouped_contexts, pregunta, role, max_response_tokens, n, temperature, limit, indexRead, typeFormat):
    results = []
    total = len(grouped_contexts)
    print('Total de requests', total)

    completo = len(grouped_contexts) + 1
    indexInit = 0
    index = completo
    if limit is not None:
        index = limit
    if indexRead is not None:
        indexInit = indexRead
        index = indexInit+limit
    print(index, indexInit)
    group_new = grouped_contexts[indexInit:index]
    i = indexInit
    for group in group_new:
        try:
            print(Colors.BLUE, len(group), 'Tokens en el grupo', i, '/', total, Colors.RESET)
            inicio_tiempo = time.time()
            response = filter_with_gpt3(group, pregunta, role, max_tokens=max_response_tokens, n=n, temperature=temperature)
            response_elem = response[0]

            if isinstance(response, list) or (typeFormat).lower == "list":
                json_list = clear_json(response_elem)
                if len(json_list) >= 1 and isinstance(json_list, list):
                    results.extend(json_list)
                else:
                    print(Colors.RED, 'Esta es la respuesta', len(response_elem), response_elem, Colors.RESET)
                    results.append(clear_json(response_elem))
            elif isinstance(response, dict) or typeFormat.lower == "json":
                results.append(json.loads(response))
            else:
                results.append(json.loads(response))

            print(f"Tiempo de procesamiento: {time.time() - inicio_tiempo} segundos")
            time.sleep(10)
        except Exception as e:
            print(f"Error procesando el grupo {i}: {e}")
        i += 1

    return results

def save_results(results, output_file='output_results'):
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    timestamp = f'{output_file}_{timestamp}.json'
  
    if os.path.exists(timestamp):
        with open(timestamp, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
    else:
        existing_data = []

    existing_data.extend(results)
    with open(timestamp, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)

    print(f"Resultados guardados en {timestamp}")

def main(config_path):
    config = load_config(config_path)
    df = leer_archivo(config["path_new"])
    processed_texts = process_dataframe(df)
    context_matrix = build_context_matrix(processed_texts, max_length=500)
    grouped_contexts = group_fragments(context_matrix, config["max_tokens"])
    role = construct_system_role(config["rules"], config["formats"])
    results = process_groups(grouped_contexts, config["pregunta"], role, config["max_response_tokens"], config["n"], config["temperature"], config.get("limit"), config.get("indexRead"), config["typeFormat"])
    if (len(results)>0):
        save_results(results)
    else: 
        print("***Revisar la conexion a OpenAI***")
