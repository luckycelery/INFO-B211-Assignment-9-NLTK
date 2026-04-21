import os
import nltk

# -------------------------------------------------------------
# DOWNLOAD REQUIRED NLTK RESOURCES
# These provide tokenizers, POS taggers, NER models, stopwords,
# and WordNet lemmatization data. They must be available before
# running any NLP pipeline.
# -------------------------------------------------------------
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk import tokenize, corpus, pos_tag
import collections
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.chunk import ne_chunk
from nltk.util import ngrams
import pandas as pd

# =====================
# FILE READING FUNCTION
# =====================
# Reads a text file into a single string.
def open_read(path):
    with open(path, "r", encoding="utf-8") as file:
        return file.read()

# =====================
# TOKENIZATION (NER)
# =====================
# Tokenizes text while preserving capitalization.
# NER models rely on capitalization to detect entities.
def tokenize_text(txt_file):
    return tokenize.word_tokenize(txt_file)

# =====================
# TOKENIZATION (ANALYSIS)
# =====================
# Lowercasing ensures consistent frequency counts.
def tokenize_lower(txt_file):
    tokens = tokenize.word_tokenize(txt_file)
    return [word.lower() for word in tokens]

# =====================
# STOP WORD REMOVAL
# =====================
# Removes common English words like "the", "and", "is".
# This highlights meaningful content words.
def remove_stop_words(tokenized_txt):
    stop_words = set(corpus.stopwords.words('english'))
    return [word for word in tokenized_txt if word not in stop_words]

# =====================
# POS TAGGING
# =====================
# Assigns grammatical roles (noun, verb, adjective, etc.)
# Required for named entity recognition.
def pos_tagging(tokens):
    return pos_tag(tokens)

# =====================
# STEMMING
# =====================
# Reduces words to crude base forms (e.g., "running" → "run").
# Included because the assignment requires it.
def stem_words(filtered_txt):
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in filtered_txt]

# =====================
# LEMMATIZATION
# =====================
# Converts words to dictionary forms (e.g., "mice" → "mouse").
# More accurate than stemming and used for token frequency analysis.
def lemmatize_words(filtered_txt):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in filtered_txt]

# =====================
# 20 MOST COMMON TOKENS
# =====================
def most_common(lemmatized_txt):
    return collections.Counter(lemmatized_txt).most_common(20)

# =====================
# NAMED ENTITY CHUNKING
# =====================
# Uses NLTK's pretrained NER model to identify PERSON, ORG, GPE, etc.
# Returns both the chunk tree and the number of detected entities.
def chunking(pos_tags):
    chunked_txt = ne_chunk(pos_tags)

    num_entities = 0
    # Walk through ALL subtrees to find labeled entities
    for subtree in chunked_txt.subtrees():
        if hasattr(subtree, "label") and subtree.label() != "S":
            num_entities += 1

    print(f"Number of named entities: {num_entities}")
    return chunked_txt, num_entities

# =====================
# MOST COMMON TRIGRAMS
# =====================
# Extracts the 5 most common 3-word sequences.
# Useful for stylistic comparison between authors.
def most_common_n_grams(lemmatized_txt):
    trigram_list = ngrams(lemmatized_txt, 3)
    return collections.Counter(trigram_list).most_common(5)

# =====================
# MAIN PROCESSING PIPELINE
# =====================
script_dir = os.path.dirname(__file__)

lovecraft_txt = open_read(os.path.join(script_dir, "RJ_Lovecraft.txt"))
rj_martin_txt = open_read(os.path.join(script_dir, "RJ_Martin.txt"))
rj_tolkein_txt = open_read(os.path.join(script_dir, "RJ_Tolkein.txt"))
martin_text = open_read(os.path.join(script_dir, "Martin.txt"))

# =====================
# PROCESS EACH TEXT
# =====================

# --- Lovecraft ---
lovecraft_tokens = tokenize_text(lovecraft_txt)
lovecraft_lower = tokenize_lower(lovecraft_txt)
lovecraft_filtered = remove_stop_words(lovecraft_lower)
lovecraft_stems = stem_words(lovecraft_filtered)
lovecraft_lemmas = lemmatize_words(lovecraft_filtered)
lovecraft_pos_tags = pos_tagging(lovecraft_tokens)
lovecraft_chunked, lovecraft_entities = chunking(lovecraft_pos_tags)
lovecraft_top_20 = most_common(lovecraft_lemmas)
lovecraft_top_trigrams = most_common_n_grams(lovecraft_lemmas)

# --- RJ Martin ---
rj_martin_tokens = tokenize_text(rj_martin_txt)
rj_martin_lower = tokenize_lower(rj_martin_txt)
rj_martin_filtered = remove_stop_words(rj_martin_lower)
rj_martin_stems = stem_words(rj_martin_filtered)
rj_martin_lemmas = lemmatize_words(rj_martin_filtered)
rj_martin_pos_tags = pos_tagging(rj_martin_tokens)
rj_martin_chunked, rj_martin_entities = chunking(rj_martin_pos_tags)
rj_martin_top_20 = most_common(rj_martin_lemmas)
rj_martin_top_trigrams = most_common_n_grams(rj_martin_lemmas)

# --- RJ Tolkien ---
rj_tolkein_tokens = tokenize_text(rj_tolkein_txt)
rj_tolkein_lower = tokenize_lower(rj_tolkein_txt)
rj_tolkein_filtered = remove_stop_words(rj_tolkein_lower)
rj_tolkein_stems = stem_words(rj_tolkein_filtered)
rj_tolkein_lemmas = lemmatize_words(rj_tolkein_filtered)
rj_tolkein_pos_tags = pos_tagging(rj_tolkein_tokens)
rj_tolkein_chunked, rj_tolkein_entities = chunking(rj_tolkein_pos_tags)
rj_tolkein_top_20 = most_common(rj_tolkein_lemmas)
rj_tolkein_top_trigrams = most_common_n_grams(rj_tolkein_lemmas)

# --- Martin (Text_4) ---
martin_tokens = tokenize_text(martin_text)
martin_lower = tokenize_lower(martin_text)
martin_filtered = remove_stop_words(martin_lower)
martin_stems = stem_words(martin_filtered)
martin_lemmas = lemmatize_words(martin_filtered)
martin_pos_tags = pos_tagging(martin_tokens)
martin_chunked, martin_entities = chunking(martin_pos_tags)
martin_top_20 = most_common(martin_lemmas)
martin_top_trigrams = most_common_n_grams(martin_lemmas)

# =====================
# COMBINED DATAFRAME FOR EASY COMPARISON
# =====================
combined_df = pd.DataFrame({
    "Text": ["Lovecraft", "RJ Martin", "RJ Tolkien", "Martin (Text_4)"],
    "Top 20 Tokens": [
        lovecraft_top_20,
        rj_martin_top_20,
        rj_tolkein_top_20,
        martin_top_20
    ],
    "Top 5 Trigrams": [
        lovecraft_top_trigrams,
        rj_martin_top_trigrams,
        rj_tolkein_top_trigrams,
        martin_top_trigrams
    ],
    "Named Entity Count": [
        lovecraft_entities,
        rj_martin_entities,
        rj_tolkein_entities,
        martin_entities
    ]
})

print("\n==================== Combined Results ====================")
print(combined_df)

combined_df.to_csv("combined_text_analysis.csv", index=False)

print("\nCombined CSV saved successfully.")
