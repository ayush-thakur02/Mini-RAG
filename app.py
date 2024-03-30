import gradio as gr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import json
from typing import List, Dict, Any

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

def preprocess(sentence: str) -> str:
    """
    Preprocesses a given sentence by converting to lowercase, tokenizing, lemmatizing, and removing stopwords.

    Parameters:
        sentence (str): The input sentence to be preprocessed.

    Returns:
        str: The preprocessed sentence.
    """
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(sentence.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def calculate_fx(sentence: str, candidates: List[str], threshold: float = 0.15) -> List[Dict[str, Any]]:
    """
    Calculates the similarity scores between the input sentence and a list of candidate sentences.

    Parameters:
        sentence (str): The input sentence.
        candidates (List[str]): List of candidate sentences.
        threshold (float, optional): Threshold value for considering a sentence similar. Defaults to 0.15.

    Returns:
        List[Dict[str, Any]]: List of dictionaries containing similar sentences and their similarity scores.
    """
    input_bits = preprocess(sentence)
    chunks = [preprocess(candidate) for candidate in candidates]
    
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([input_bits] + chunks)
    
    f_scores = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    
    similar_chunks = []
    for i, score in enumerate(f_scores):
        if score >= threshold:
            similar_chunks.append({"sentence": candidates[i], "f(score)": round(score, 4)})
    
    return similar_chunks

def read_sentences_from_file(file_location: str) -> List[str]:
    """
    Reads sentences from a text file located at the given location.

    Parameters:
        file_location (str): Location of the text file.

    Returns:
        List[str]: List of sentences read from the file.
    """
    with open(file_location, 'r') as file:
        text = file.read().replace('\n', ' ')
        sentences = [sentence.strip() for sentence in text.split('.') if sentence.strip()]
    return sentences

def fetch_vectors(file: Any, sentence: str) -> str:
    """
    Fetches similar sentences from a text file for a given input sentence.

    Parameters:
        file (Any): File uploaded by the user.
        sentence (str): Input sentence.

    Returns:
        str: JSON string containing similar sentences and their similarity scores.
    """
    file_location = file.name
    chunks = read_sentences_from_file(file_location)
    similar_chunks = calculate_fx(sentence, chunks, threshold=0.15)
    return json.dumps(similar_chunks, indent=4)

# Interface
file_uploader = gr.File(label="Upload a .txt file")
text_input = gr.Textbox(label="Enter question")
output_text = gr.Textbox(label="Output")

iface = gr.Interface(
    fn=fetch_vectors,
    inputs=[file_uploader, text_input],
    outputs=output_text,
    title="Minimal RAG - For QA (Super Fast/Modeless)",
    description="Fastest Minimal Rag for Question Answer, calculating cosine similarities and vectorizing using scikit-learn's TfidfVectorizer."
)

iface.launch(debug=True)
