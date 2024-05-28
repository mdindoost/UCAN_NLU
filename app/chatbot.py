import json
import spacy
from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import difflib
import tensorflow_hub as hub
import logging
import os

# Create a directory for logs if it doesn't exist
log_dir = '/home/mohammad/UCAN/app/logs'  # Change this to your desired directory
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, 'chatbot.log')

# Set up logging to file and console
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(log_file_path),
                        logging.StreamHandler()
                    ])

# Load the spaCy model with word vectors
nlp = spacy.load('en_core_web_lg')  # or 'en_core_web_md'

def load_knowledge_base(file_path='knowledge_base.json'):
    """Load the knowledge base from a JSON file."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

knowledge_base = load_knowledge_base()

def preprocess_text(text):
    """Preprocess text by lowercasing and lemmatizing."""
    doc = nlp(text.lower())
    return ' '.join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

def extract_keywords(text):
    """Extract keywords from text using spaCy."""
    doc = nlp(text.lower())
    return set(token.lemma_ for token in doc if not token.is_stop and not token.is_punct)

# Load BERT model and tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def embed_text_bert(text):
    """Embed text using BERT."""
    inputs = bert_tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    outputs = bert_model(**inputs)
    cls_embeddings = outputs.last_hidden_state[:, 0, :]
    return cls_embeddings

# Load RoBERTa model and tokenizer
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta_model = RobertaModel.from_pretrained('roberta-base')

def embed_text_roberta(text):
    """Embed text using RoBERTa."""
    inputs = roberta_tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    outputs = roberta_model(**inputs)
    cls_embeddings = outputs.last_hidden_state[:, 0, :]
    return cls_embeddings

# Load Universal Sentence Encoder
use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

def embed_text_use(text):
    """Embed text using Universal Sentence Encoder."""
    return use_model([text]).numpy()

def get_best_match_spacy(user_input, knowledge_base):
    """Get the best matching answer from the knowledge base using spaCy similarity."""
    user_doc = nlp(preprocess_text(user_input))
    best_match = None
    best_score = 0.0

    for faq in knowledge_base['faqs']:
        faq_doc = nlp(preprocess_text(faq['question']))
        similarity = user_doc.similarity(faq_doc)
        if similarity > best_score:
            best_score = similarity
            best_match = faq['answer']

    return best_match, best_score

def get_best_match_bert(user_input, knowledge_base):
    """Get the best matching answer from the knowledge base using BERT embeddings."""
    user_embedding = embed_text_bert(preprocess_text(user_input))
    best_match = None
    best_score = 0.0

    for faq in knowledge_base['faqs']:
        faq_embedding = embed_text_bert(preprocess_text(faq['question']))
        similarity = cosine_similarity(user_embedding.detach().numpy(), faq_embedding.detach().numpy())[0][0]
        if similarity > best_score:
            best_score = similarity
            best_match = faq['answer']

    return best_match, best_score

def get_best_match_roberta(user_input, knowledge_base):
    """Get the best matching answer from the knowledge base using RoBERTa embeddings."""
    user_embedding = embed_text_roberta(preprocess_text(user_input))
    best_match = None
    best_score = 0.0

    for faq in knowledge_base['faqs']:
        faq_embedding = embed_text_roberta(preprocess_text(faq['question']))
        similarity = cosine_similarity(user_embedding.detach().numpy(), faq_embedding.detach().numpy())[0][0]
        if similarity > best_score:
            best_score = similarity
            best_match = faq['answer']

    return best_match, best_score

def get_best_match_use(user_input, knowledge_base):
    """Get the best matching answer from the knowledge base using Universal Sentence Encoder."""
    user_embedding = embed_text_use(user_input)
    best_match = None
    best_score = 0.0

    for faq in knowledge_base['faqs']:
        faq_embedding = embed_text_use(faq['question'])
        similarity = cosine_similarity(user_embedding, faq_embedding)[0][0]
        if similarity > best_score:
            best_score = similarity
            best_match = faq['answer']

    return best_match, best_score

def fuzzy_match(user_input, knowledge_base):
    """Get the best matching answer using fuzzy matching."""
    best_match = None
    best_score = 0.0

    for faq in knowledge_base['faqs']:
        similarity = difflib.SequenceMatcher(None, user_input.lower(), faq['question'].lower()).ratio()
        if similarity > best_score:
            best_score = similarity
            best_match = faq['answer']

    return best_match, best_score

def keywords_in_answer(keywords, answer):
    """Check if all keywords are present in the answer."""
    answer_keywords = extract_keywords(answer)
    return keywords.issubset(answer_keywords)

def get_best_match(user_input, knowledge_base):
    """Combine spaCy, BERT, RoBERTa, USE, and fuzzy matching with a heuristic approach."""
    match_spacy, score_spacy = get_best_match_spacy(user_input, knowledge_base)
    match_bert, score_bert = get_best_match_bert(user_input, knowledge_base)
    match_roberta, score_roberta = get_best_match_roberta(user_input, knowledge_base)
    match_use, score_use = get_best_match_use(user_input, knowledge_base)
    match_fuzzy, score_fuzzy = fuzzy_match(user_input, knowledge_base)

    # Extract keywords from the user input
    keywords = extract_keywords(user_input)

    logging.info(f"User input: {user_input}")
    logging.info(f"Keywords: {keywords}")
    logging.info(f"spaCy score: {score_spacy}, match: {match_spacy}")
    logging.info(f"BERT score: {score_bert}, match: {match_bert}")
    logging.info(f"RoBERTa score: {score_roberta}, match: {match_roberta}")
    logging.info(f"USE score: {score_use}, match: {match_use}")
    logging.info(f"Fuzzy score: {score_fuzzy}, match: {match_fuzzy}")

    # Combine the scores (weighted average or highest score)
    scores = {
        'spacy': score_spacy * 1.0,
        'bert': score_bert * 0.9,
        'roberta': score_roberta * 1.1,
        'use': score_use * 1.1,
        'fuzzy': score_fuzzy * 0.8
    }

    best_match = max(scores, key=scores.get)

    # Heuristic: Prioritize models if the score is above a threshold and it has all keywords
    if score_spacy > 0.75 and keywords_in_answer(keywords, match_spacy):
        return match_spacy
    elif score_bert > 0.75 and keywords_in_answer(keywords, match_bert):
        return match_bert
    elif score_roberta > 0.75 and keywords_in_answer(keywords, match_roberta):
        return match_roberta
    elif score_use > 0.75 and keywords_in_answer(keywords, match_use):
        return match_use
    elif score_fuzzy > 0.75 and keywords_in_answer(keywords, match_fuzzy):
        return match_fuzzy
    else:
        # Fallback to the highest score with keyword check
        candidates = [(match_spacy, score_spacy), (match_bert, score_bert), (match_roberta, score_roberta), (match_use, score_use), (match_fuzzy, score_fuzzy)]
        candidates_with_keywords = [(match, score) for match, score in candidates if keywords_in_answer(keywords, match)]
        if candidates_with_keywords:
            best_match, _ = max(candidates_with_keywords, key=lambda x: x[1])
        else:
            best_match, _ = max(candidates, key=lambda x: x[1])
        return best_match

def chatbot_response(user_input):
    """Generate a response for the user input by finding the best match in the knowledge base."""
    logging.info(f"Received question: {user_input}")
    response = get_best_match(user_input, knowledge_base)
    logging.info(f"Response: {response}")
    return response

if __name__ == '__main__':
    # Test the chatbot with a sample question
    sample_question = "What are the benefits for Research Employees?"
    print(chatbot_response(sample_question))
