import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Cargar modelos de Hugging Face
model_id = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

# Cargar el modelo de spaCy
nlp = spacy.load("en_core_web_sm")


