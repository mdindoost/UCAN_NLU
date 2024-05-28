import json
import spacy

# Load the spaCy model
nlp = spacy.load('en_core_web_sm')

def load_knowledge_base(file_path='knowledge_base.json'):
    """Load the knowledge base from a JSON file."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

knowledge_base = load_knowledge_base()

def process_input(user_input):
    """Process the user input using spaCy to get lemmas."""
    doc = nlp(user_input)
    return ' '.join([token.lemma_ for token in doc])

def get_answer_from_json(question, knowledge_base):
    """Get the answer from the JSON knowledge base by matching the processed question."""
    for faq in knowledge_base['faqs']:
        if faq['question'].lower() == question.lower():
            return faq['answer']
    return "Sorry, I do not know the answer to that question."

def chatbot_response(user_input):
    """Generate a response for the user input by processing it and searching the knowledge base."""
    processed_input = process_input(user_input)
    return get_answer_from_json(processed_input, knowledge_base)
