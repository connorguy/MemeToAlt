import os
import re
import string
import logging

import nltk
import spacy
from nltk import WordNetLemmatizer

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s:: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
log = logging.getLogger(__name__)


nltk.download('words')
nltk.download('wordnet')
nltk.download('omw-1.4')

nlp = spacy.load("en_core_web_lg")

REMOVE_PUNCTUATION = True  # Changing this can create different results in remove_uncorrelated_text
REMOVE_UNCORRELATED_WORDS = True
REMOVE_NON_ENGLISH_WORDS = False


def clean_text(text: []) -> []:
    for index, val in enumerate(text):
        # print("Input:: ",val)
        if REMOVE_PUNCTUATION:
            # text[index] = remove_punctuation(val)
            text[index] = val.lstrip(string.punctuation)
            # print("Punc: ",text[index])

        if REMOVE_NON_ENGLISH_WORDS:
            # text[index] = remove_non_english_words(val)
            print("Eng: ",text[index])

    if REMOVE_UNCORRELATED_WORDS:
        text = remove_uncorrelated_text(text)
    return text


def remove_non_english_words(text) -> str:
    lemmatizer = WordNetLemmatizer()
    words = set(nltk.corpus.words.words())
    replace_words = set(w for w in nltk.wordpunct_tokenize(text) if lemmatizer.lemmatize(w.lower()) not in words)
    for t in replace_words:
        if t not in string.punctuation:
            text = text.replace(t, ' ').strip()
    return text


def remove_punctuation(text) -> str:
    return re.sub(r'[^\w\s]', '', text)


def remove_uncorrelated_text(input: []) -> []:
    cleaned_input = []
    for t in input:
        if t.strip() == '':
            continue
        score = nlp(t).similarity(nlp(input[0]))  # this could be better
        log.info("Score: %s :: %s" % (score, t))
        if score > 0.15:
            cleaned_input.append(t)
    return cleaned_input


if __name__ == '__main__':
    text_input = []
    text_input.append("My SOURCES?")
    text_input.append("ALIENS")
    text_input.append("H")
    text_input.append("Inedeneialog_Ial")
    # text_input.append("history channel logo 2020")
    # remove_uncorrelated_text(text_input)

    print("Iitial:: ", text_input)
    out = clean_text(text_input)
    print("FINAL:: ",out)
