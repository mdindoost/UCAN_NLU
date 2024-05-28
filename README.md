In this project I tried to combine 5 different methods. right now it has only knwledge of NJIT UCAN MOA document. LOL

1. SpaCy
Overview:

SpaCy is an open-source library for Natural Language Processing (NLP) in Python.
It is designed for production use and provides a range of NLP features.
Role in Chatbot:

Text Preprocessing: SpaCy is used to preprocess text by lowercasing and lemmatizing the input. This helps standardize the text and reduce variations.
Keyword Extraction: SpaCy is used to extract keywords from both the user query and the knowledge base answers. This helps in ensuring that the responses are contextually relevant.
Similarity Measurement: SpaCy calculates the similarity between the processed user query and each question in the knowledge base. The similarity score is used to find the most relevant answer.

2. BERT (Bidirectional Encoder Representations from Transformers)
Overview:

BERT is a pre-trained transformer-based model designed by Google.
It provides contextual embeddings for words, capturing the meaning based on surrounding words.
Role in Chatbot:

Text Embedding: BERT is used to generate embeddings for the user query and the questions in the knowledge base. These embeddings capture the contextual meaning of the text.
Similarity Measurement: Cosine similarity is calculated between the BERT embeddings of the user query and each question in the knowledge base to find the best match.

3. Universal Sentence Encoder (USE)
Overview:

The Universal Sentence Encoder (USE) is a model by Google that encodes text into high-dimensional vectors.
It is optimized for encoding sentences and longer text passages into fixed-length vectors.
Role in Chatbot:

Text Embedding: USE generates embeddings for the user query and the questions in the knowledge base. These embeddings capture the semantic meaning of the text.
Similarity Measurement: Cosine similarity is calculated between the USE embeddings of the user query and each question in the knowledge base to find the best match.

4. Fuzzy Matching (Using difflib)
Overview:

Fuzzy matching compares two strings and determines their similarity based on the sequence of characters.
It is useful for handling minor variations and typos in text.
Role in Chatbot:

Similarity Measurement: Fuzzy matching calculates the similarity between the user query and each question in the knowledge base based on character sequences. This helps in identifying relevant questions even if there are slight variations in wording.

5. Keyword Matching
Overview:

Keyword matching ensures that the key concepts from the user query are present in the selected answer.
This helps in verifying that the response is contextually relevant.
Role in Chatbot:

Keyword Verification: After identifying potential matches using the various models, keyword matching ensures that the selected answer contains the relevant keywords from the user query.
