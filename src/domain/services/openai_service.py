import openai

def filter_with_gpt3(texts, question, role, model_name='gpt-3.5-turbo', max_tokens=150, n=1, temperature=0.7):
    openai.api_key = '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'
    
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

def construct_system_role(rules, formats):
    role_content = "Eres un asistente experto en responder preguntas basadas en contexto y de forma concisa. "
    if rules:
        role_content += "Reglas: " + ", ".join(rules) + ". "
    if formats:
        role_content += "Formatos de respuesta: " + ", ".join(formats) + "."
    return role_content
