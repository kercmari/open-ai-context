from gensim.utils import simple_preprocess
from ..models.nlp_models import nlp, tokenizer

def process_text(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop]
    return " ".join(tokens)

def gensim_preprocess(text):
    tokens = simple_preprocess(text, deacc=True)  # deacc=True elimina acentos
    return tokens

def split_into_fragments(text, max_length=500):
    tokens = tokenizer(text)["input_ids"]
    return [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]

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
