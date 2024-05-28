import json
import spacy
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import difflib
import tensorflow_hub as hub

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
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def embed_text(text):
    """Embed text using BERT."""
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
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
    user_embedding = embed_text(preprocess_text(user_input))
    best_match = None
    best_score = 0.0

    for faq in knowledge_base['faqs']:
        faq_embedding = embed_text(preprocess_text(faq['question']))
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
    """Combine spaCy, BERT, USE, and fuzzy matching with a heuristic approach."""
    match_spacy, score_spacy = get_best_match_spacy(user_input, knowledge_base)
    match_bert, score_bert = get_best_match_bert(user_input, knowledge_base)
    match_use, score_use = get_best_match_use(user_input, knowledge_base)
    match_fuzzy, score_fuzzy = fuzzy_match(user_input, knowledge_base)

    # Extract keywords from the user input
    keywords = extract_keywords(user_input)

    print("score_spacy = ", score_spacy)
    print("match_spacy = ", match_spacy)

    print("**********************************************")
    print("score_bert = ", score_bert)
    print("match_bert = ", match_bert)

    print("**********************************************")
    print("score_use = ", score_use)
    print("match_use = ", match_use)

    print("**********************************************")
    print("score_fuzzy = ", score_fuzzy)
    print("match_fuzzy = ", match_fuzzy)

    # Combine the scores (weighted average or highest score)
    scores = {
        'spacy': score_spacy * 1.0,  # Adjust weight as necessary
        'bert': score_bert * 0.9,    # Adjust weight as necessary
        'use': score_use * 1.1,      # Adjust weight as necessary
        'fuzzy': score_fuzzy * 0.8   # Adjust weight as necessary
    }

    best_match = max(scores, key=scores.get)

    # Heuristic: Prioritize models if the score is above a threshold and it has all keywords
    if score_spacy > 0.75 and keywords_in_answer(keywords, match_spacy):
        return match_spacy
    elif score_bert > 0.75 and keywords_in_answer(keywords, match_bert):
        return match_bert
    elif score_use > 0.75 and keywords_in_answer(keywords, match_use):
        return match_use
    elif score_fuzzy > 0.75 and keywords_in_answer(keywords, match_fuzzy):
        return match_fuzzy
    else:
        # Fallback to the highest score with keyword check
        candidates = [(match_spacy, score_spacy), (match_bert, score_bert), (match_use, score_use), (match_fuzzy, score_fuzzy)]
        candidates_with_keywords = [(match, score) for match, score in candidates if keywords_in_answer(keywords, match)]
        if candidates_with_keywords:
            best_match, _ = max(candidates_with_keywords, key=lambda x: x[1])
        else:
            best_match, _ = max(candidates, key=lambda x: x[1])
        return best_match

def chatbot_response(user_input):
    """Generate a response for the user input by finding the best match in the knowledge base."""
    return get_best_match(user_input, knowledge_base)

if __name__ == '__main__':
    # Test the chatbot with a sample question
    sample_question = "What are the benefits for Research Employees?"
    print(chatbot_response(sample_question))
