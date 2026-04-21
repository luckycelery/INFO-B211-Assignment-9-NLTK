import os
import nltk
import string

# -------------------------------------------------------------
# DOWNLOAD REQUIRED NLTK RESOURCES
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
def open_read(path):
    with open(path, "r", encoding="utf-8") as file:
        return file.read()

# =====================
# TOKENIZATION (NER)
# =====================
def tokenize_text(txt_file):
    # Tokenize the source text into individual words (tokens)
    return tokenize.word_tokenize(txt_file)

# =====================
# TOKENIZATION (ANALYSIS)
# =====================
def tokenize_lower(txt_file):
    # Tokenize the source text into individual words (tokens) and convert to lowercase for analysis
    tokens = tokenize.word_tokenize(txt_file)
    return [word.lower() for word in tokens]

# =====================
# REMOVE PUNCTUATION
# =====================
# Removes tokens that ARE punctuation
def remove_punctuation(tokens):
    return [word for word in tokens if word not in string.punctuation]

# =====================
# STOP WORD REMOVAL
# =====================
def remove_stop_words(tokenized_txt):
    # Remove common words (stop words) that do not add much meaning to the text
    stop_words = set(corpus.stopwords.words('english'))
    return [word for word in tokenized_txt if word not in stop_words]

# =====================
# POS TAGGING
# =====================
def pos_tagging(tokens):
    # Assign part-of-speech tags to each token to understand their grammatical roles
    return pos_tag(tokens)

# =====================
# STEMMING
# =====================
def stem_words(filtered_txt):
    stemmer = PorterStemmer()
    # Reduce words to their root form (stemming) to group similar words together
    return [stemmer.stem(word) for word in filtered_txt]

# =====================
# LEMMATIZATION
# =====================
def lemmatize_words(filtered_txt):
    lemmatizer = WordNetLemmatizer()
    # Reduce words to their base or dictionary form (lemmatization) for more accurate grouping of similar words
    return [lemmatizer.lemmatize(word) for word in filtered_txt]

# =====================
# 20 MOST COMMON TOKENS
# =====================
def most_common(lemmatized_txt):
    # Count the frequency of each token and return the 20 most common tokens along with their counts
    return collections.Counter(lemmatized_txt).most_common(20)

# =====================
# NAMED ENTITY CHUNKING
# =====================
def chunking(pos_tags):
    # Identify and classify named entities (like people, organizations, locations) in the text
    chunked_txt = ne_chunk(pos_tags)
    # Count the number of named entities identified in the text
    num_entities = 0
    for subtree in chunked_txt.subtrees():
        # Only count subtrees that are labeled as named entities (e.g., PERSON, ORGANIZATION, GPE) and exclude the top-level S (sentence) label
        if hasattr(subtree, "label") and subtree.label() != "S":
            num_entities += 1

    print(f"Number of named entities: {num_entities}")
    return chunked_txt, num_entities

# =====================
# MOST COMMON TRIGRAMS
# =====================
def most_common_n_grams(lemmatized_txt):
    # Generate trigrams (sequences of 3 consecutive tokens) from the lemmatized text and return the 5 most common trigrams along with their counts
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
lovecraft_no_punct = remove_punctuation(lovecraft_lower)
lovecraft_filtered = remove_stop_words(lovecraft_no_punct)
lovecraft_stems = stem_words(lovecraft_filtered)
lovecraft_lemmas = lemmatize_words(lovecraft_filtered)
lovecraft_pos_tags = pos_tagging(lovecraft_tokens)
lovecraft_chunked, lovecraft_entities = chunking(lovecraft_pos_tags)
lovecraft_top_20 = most_common(lovecraft_lemmas)
lovecraft_top_trigrams = most_common_n_grams(lovecraft_lemmas)

# --- RJ Martin ---
rj_martin_tokens = tokenize_text(rj_martin_txt)
rj_martin_lower = tokenize_lower(rj_martin_txt)
rj_martin_no_punct = remove_punctuation(rj_martin_lower)
rj_martin_filtered = remove_stop_words(rj_martin_no_punct)
rj_martin_stems = stem_words(rj_martin_filtered)
rj_martin_lemmas = lemmatize_words(rj_martin_filtered)
rj_martin_pos_tags = pos_tagging(rj_martin_tokens)
rj_martin_chunked, rj_martin_entities = chunking(rj_martin_pos_tags)
rj_martin_top_20 = most_common(rj_martin_lemmas)
rj_martin_top_trigrams = most_common_n_grams(rj_martin_lemmas)

# --- RJ Tolkien ---
rj_tolkein_tokens = tokenize_text(rj_tolkein_txt)
rj_tolkein_lower = tokenize_lower(rj_tolkein_txt)
rj_tolkein_no_punct = remove_punctuation(rj_tolkein_lower)
rj_tolkein_filtered = remove_stop_words(rj_tolkein_no_punct)
rj_tolkein_stems = stem_words(rj_tolkein_filtered)
rj_tolkein_lemmas = lemmatize_words(rj_tolkein_filtered)
rj_tolkein_pos_tags = pos_tagging(rj_tolkein_tokens)
rj_tolkein_chunked, rj_tolkein_entities = chunking(rj_tolkein_pos_tags)
rj_tolkein_top_20 = most_common(rj_tolkein_lemmas)
rj_tolkein_top_trigrams = most_common_n_grams(rj_tolkein_lemmas)

# --- Martin (Text_4) ---
martin_tokens = tokenize_text(martin_text)
martin_lower = tokenize_lower(martin_text)
martin_no_punct = remove_punctuation(martin_lower)
martin_filtered = remove_stop_words(martin_no_punct)
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
