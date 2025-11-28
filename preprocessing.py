# -------------------- IMPORTS --------------------
import re
import string
import unicodedata
import contractions
import emoji
import spacy
from functools import lru_cache
from spacy.lang.en.stop_words import STOP_WORDS as spacy_stopwords
from spellchecker import SpellChecker

# ---------------- NLP LOADER ----------------
@lru_cache(maxsize=1)
def get_nlp():
    """
    Load the SpaCy NLP model with caching for speed.
    Disables parser and NER for faster processing since we only need tokenization/lemmatization.
    lru_cache ensures the model is loaded only once.
    """
    return spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])

# ---------------- STOPWORDS ----------------
# Words we want to keep even though they are common stopwords
keep_stopwords = set([
    'again','against','all','any','aren','below','couldn','didn','doesn','down','few','further',
    'hadn', 'hasn','haven','more','most','mustn','needn','no','nor','not','out','over','own',
    'shan','shouldn','under','up','very','wasn','weren','won','too','until','at','by','of','off','on'
])

# Custom stopwords: remove all standard spacy stopwords except the ones we want to keep
custom_stopwords = {w for w in spacy_stopwords if w not in keep_stopwords}

# ---------------- SPELLCHECKER ----------------
# SpellChecker instance with edit distance=1 for minor spelling corrections
spell = SpellChecker(distance=1)

# ---------------- CLEANING ----------------
def clean_text(text):
    """
    Clean input text by:
    - Lowercasing
    - Expanding contractions
    - Removing HTML tags, URLs, emails, numbers
    - Removing emojis
    - Normalizing unicode (removing accents)
    - Removing punctuation
    - Removing extra whitespace
    Returns cleaned string.
    """
    text = text.lower()                              # lowercase
    text = contractions.fix(text)                    # expand contractions like "don't" -> "do not"
    text = re.sub(r"<.*?>", " ", text)              # remove HTML tags
    text = re.sub(r"http\S+|www\S+", " ", text)     # remove URLs
    text = re.sub(r"\S+@\S+\.\S+", " ", text)       # remove emails
    text = re.sub(r"\d+", " ", text)                # remove numbers
    text = emoji.replace_emoji(text, replace="")    # remove emojis
    text = unicodedata.normalize("NFKD", text)      # normalize unicode characters
    text = text.encode("ascii", "ignore").decode("utf-8", "ignore")  # remove non-ASCII characters
    text = text.translate(str.maketrans("", "", string.punctuation))  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()        # remove extra spaces
    return text

# ---------------- MAIN PIPELINE For ML----------------
def preprocess_new_texts(texts):
    """
    Full preprocessing pipeline for classical ML models:
    - Clean text
    - Tokenize and lemmatize using SpaCy
    - Remove stopwords
    - Correct spelling
    Returns a list of preprocessed strings.
    """
    nlp = get_nlp()
    cleaned = [clean_text(t) for t in texts]  # Clean each text
    final_texts = []

    # Use SpaCy pipe for batch processing (n_process=1 for short texts)
    for doc in nlp.pipe(cleaned, batch_size=1, n_process=1):
        # Lemmatize and remove stopwords
        tokens = [
            token.lemma_.lower()       # lemmatize and lowercase
            for token in doc
            if token.is_alpha and token.text not in custom_stopwords  # keep only alphabetic tokens not in stopwords
        ]

        # Spell check: correct misspelled words
        tokens_corrected = [
            spell.correction(token) if token not in spell.word_frequency else token
            for token in tokens
        ]

        # Join tokens back into a single string
        final_texts.append(" ".join(tokens_corrected))

    return final_texts

# ---------------- CLEANING FOR NN ----------------
def clean_text(text):
    """
    Simplified cleaning function for neural networks:
    - Lowercase
    - Expand contractions
    - Remove HTML, URLs, emails, non-ASCII chars, extra spaces
    """
    text = text.lower()                              # lowercase
    text = contractions.fix(text)                    # expand contractions
    text = re.sub(r"<.*?>", " ", text)              # remove HTML tags
    text = re.sub(r"http\S+|www\S+", " ", text)     # remove URLs
    text = re.sub(r"\S+@\S+\.\S+", " ", text)       # remove emails
    text = unicodedata.normalize("NFKD", text)      # normalize unicode
    text = text.encode("ascii", "ignore").decode("utf-8", "ignore")  # remove non-ASCII
    text = re.sub(r"\s+", " ", text).strip()        # remove extra spaces
    return text
