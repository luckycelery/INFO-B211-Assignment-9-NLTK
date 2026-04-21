# INFO-B211-Assignment-9-NLTK
# NLP Comparative Text Analysis Project  
### Using Tokenization, Stemming, Lemmatization, NER, and Trigram Analysis

---

# 1. **Analysis of the Four Texts (Assignment Questions Answered)**

This project applies Natural Language Processing (NLP) techniques to three primary texts (Text_1, Text_2, Text_3) and a fourth unknown text (Text_4). The goal is to extract structure from unstructured data and use linguistic patterns to infer subject matter and potential authorship.

After implementing improved punctuation removal, the combined results table is:

| Text | Top 20 Tokens | Top 5 Trigrams | Named Entity Count |
|------|---------------|----------------|--------------------|
| **Lovecraft** | [('cosmic', 7), ('eldritch', 5), ('amidst', 5), ('juliet', 5), ('romeo', 5), ('ancient', 4), ('fate', 4), ('mortal', 4), ('love', 4), ('verona', 3), ('whisper', 3), ('forbidden', 3), ('tale', 3), ('bore', 3), ('embrace', 3), ('force', 3), ('city', 2), ('echoed', 2), ('pact', 2), ('tragedy', 2)] | [(('\ufeffin', 'hallowed', 'city'), 1), (('hallowed', 'city', 'verona'), 1), (('city', 'verona', 'ancient'), 1), (('verona', 'ancient', 'stone'), 1), (('ancient', 'stone', 'echoed'), 1)] | **15** |
| **RJ Martin** | [('house', 5), ('amidst', 5), ('romeo', 5), ('juliet', 4), ('love', 4), ('passion', 3), ('fate', 3), ('city', 2), ('verona', 2), ('capulet', 2), ('montague', 2), ('star-crossed', 2), ('hall', 2), ('twisted', 2), ('ancient', 2), ('thorn', 2), ('secret', 2), ('garden', 2), ('steeped', 2), ('tragedy', 2)] | [(('\ufeffin', 'sprawling', 'city'), 1), (('sprawling', 'city', 'verona'), 1), (('city', 'verona', 'towering'), 1), (('verona', 'towering', 'wall'), 1), (('towering', 'wall', 'house'), 1)] | **14** |
| **RJ Tolkien** | [('juliet', 7), ('verona', 5), ('amidst', 5), ('romeo', 5), ('land', 4), ('love', 4), ('woven', 3), ('fate', 3), ('upon', 3), ('discord', 3), ('tragic', 3), ('eternal', 3), ('despair', 3), ('among', 2), ('ancient', 2), ('tale', 2), ('thread', 2), ('tapestry', 2), ('fair', 2), ("'s", 2)] | [(('\ufeffin', 'verdant', 'land'), 1), (('verdant', 'land', 'verona'), 1), (('land', 'verona', 'nestled'), 1), (('verona', 'nestled', 'among'), 1), (('nestled', 'among', 'rolling'), 1)] | **23** |
| **Martin (Text_4)** | [('’', 58), ("''", 45), ('``', 43), ('aldric', 32), ('eye', 29), ('“', 22), ('”', 22), ('cold', 20), ('alysanne', 19), ('like', 18), ('torran', 17), ('could', 17), ('shadow', 15), ('voice', 15), ('said', 14), ('sword', 13), ('would', 12), ('merek', 12), ('heart', 11), ('house', 10)] | [(("''", 'aldric', '’'), 5), (('”', 'torran', '’'), 3), (('could', 'feel', 'eye'), 3), (('keep', 'ser', 'aldric'), 2), (('”', 'torran', 'said'), 2)] | **179** |

---

## 1.1 **Subject of the First Three Texts**

Across Lovecraft, RJ Martin, and RJ Tolkien, the most frequent tokens and trigrams consistently reference:

- romeo  
- juliet  
- verona  
- love  
- fate  
- ancient  
- tale  

These patterns indicate that all three texts are stylistically distinct retellings of **Romeo and Juliet**, each filtered through the voice of a different author:

- **Lovecraft** emphasizes cosmic dread and fatalism.  
- **RJ Martin** frames the story through political tension and house conflict.  
- **RJ Tolkien** presents a mythic, fate‑woven, high‑fantasy interpretation.

---

## 1.2 **Authorship Analysis of Text_4**

Text_4 diverges sharply from the first three texts:

- No references to Romeo, Juliet, or Verona  
- Heavy use of character names (aldric, allysanne, torran)  
- Frequent introspective and sensory verbs (could, feel, said, voice)  
- Dialogue‑heavy punctuation (curly quotes, apostrophes, double‑quotes)

The tone is:

- grounded  
- gritty  
- character‑driven  
- political‑fantasy oriented  

### **Conclusion**  
Based on vocabulary, trigrams, and narrative tone, **Text_4 most closely resembles RJ Martin’s style**.  
It is highly likely that **RJ Martin is the author of Text_4**.

---

# 2. **Purpose of the Project**

The purpose of this project is to demonstrate how Natural Language Processing (NLP) techniques can extract structure, meaning, and stylistic patterns from unstructured text. By applying tokenization, stemming, lemmatization, named entity recognition, and trigram analysis, we can:

- identify thematic content  
- compare writing styles  
- quantify linguistic features  
- infer authorship  

This project highlights how computational methods can support literary analysis, authorship attribution, and text classification.

---

# 3. **NLP Pipeline Overview**

The project uses a modular, step‑by‑step NLP pipeline:

1. Load text  
2. Tokenize  
   - original‑case tokens for NER  
   - lowercase tokens for frequency analysis  
3. Remove punctuation
4. Remove stopwords  
5. Stem tokens  
6. Lemmatize tokens  
7. POS‑tag original tokens  
8. Run Named Entity Recognition  
9. Compute top 20 tokens  
10. Compute top 5 trigrams  
11. Aggregate results into a combined DataFrame  
12. Export to CSV  

---

# 4. **Function Documentation**

## 4.1 `open_read(path)`
**Input:** file path  
**Output:** raw text string  
**Purpose:** Reads text files into memory.

---

## 4.2 `tokenize_text(txt_file)`
**Input:** raw text  
**Output:** list of tokens (original case)  
**Purpose:** Used for POS tagging and NER.

---

## 4.3 `tokenize_lower(txt_file)`
**Input:** raw text  
**Output:** list of lowercase tokens  
**Purpose:** Used for frequency analysis.

---

## 4.4 `remove_punctuation(tokens)`
**Input:** list of lowercase tokens  
**Output:** list of tokens with punctuation removed  
**Purpose:**  
Many texts contain punctuation tokens that distort frequency counts and trigram analysis. NLTK often treats punctuation as standalone tokens, including multi‑character sequences such as `''`, `````, `…`, and curly quotes.  
This function removes **any token containing at least one punctuation character**, ensuring that only meaningful words remain for downstream analysis.

---

## 4.5 `remove_stop_words(tokenized_txt)`
**Input:** token list  
**Output:** token list without stopwords  
**Purpose:** Removes common English words to highlight meaningful content.

---

## 4.6 `pos_tagging(tokens)`
**Input:** token list  
**Output:** list of (token, POS tag) pairs  
**Purpose:** Required for NER.

---

## 4.7 `stem_words(filtered_txt)`
**Input:** token list  
**Output:** list of stems  
**Purpose:** Assignment requirement; not used for final metrics.

---

## 4.8 `lemmatize_words(filtered_txt)`
**Input:** token list  
**Output:** list of lemmas  
**Purpose:** Used for top token and trigram analysis.

---

## 4.9 `most_common(lemmatized_txt)`
**Input:** list of lemmas  
**Output:** top 20 `(token, count)` pairs  
**Purpose:** Identifies key vocabulary.

---

## 4.10 `chunking(pos_tags)`
**Input:** POS‑tagged tokens  
**Output:** NER chunk tree and named entity count  
**Purpose:** Extracts real‑world entities (people, places, organizations).

---

## 4.11 `most_common_n_grams(lemmatized_txt)`
**Input:** list of lemmas  
**Output:** top 5 trigrams  
**Purpose:** Captures stylistic and thematic patterns.

---

# 5. **Limitations**

- NLTK’s NER model is limited compared to modern transformer‑based models.  
- The analysis focuses on surface‑level features, not semantic meaning.  
- Some Unicode punctuation (e.g., curly quotes) is not included in `string.punctuation` and may still appear.  
- Functional design is simple for this assignment but not ideal for large‑scale systems.

---

# 6. **Output**

The script produces:

- A printed combined results table  
- A CSV file: **`combined_text_analysis.csv`**

This file contains all metrics for all four texts in a single, easy‑to‑compare format.

---

# 7. **Conclusion**

Using NLP techniques, we determined that:

- All three primary texts are stylistically distinct retellings of **Romeo and Juliet**.  
- Text_4 is most stylistically similar to **RJ Martin**, suggesting he is the likely author.

This project demonstrates how computational linguistics can support literary analysis and authorship attribution.


