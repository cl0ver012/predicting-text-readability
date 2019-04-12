"""
Functions for creation of commonly used text readability features.

All functions take a pandas dataframe as an input. The dataframe contains texts in the Text column.
The functions will calculate the feature for every text in the dataframe, creating a column for the feature.
"""
import pandas as pd

# for tokenization
import spacy


# the following spacy model has to be downloaded
SPACY_MODEL = "en_core_web_sm"


def _get_words(x):
    words = [token.text for token in x if token.is_punct != True]
    return words


def words_and_sentences(df):
    """
    Uses spacy to find number of words and sentences for each text.
    
    Added features:
    Tokens: all tokens in the text
    Words: all words in the text
    Sentences: all sentences in the text
    N_words: number of words in the text
    N_sentences: number of sentences in the text
    
    :param: the dataframe with the dataset
    :returns: the dataframe with added features
    """
    
    # load spacy model
    nlp = spacy.load(SPACY_MODEL, parser=False, entity=False)
    
    # get tokens
    df['Tokens'] = df['Text'].apply(lambda x: nlp(x))
    
    # get words
    df['Words'] = df['Tokens'].apply(lambda x: _get_words(x))
    
    # get sentences
    df['Sentences'] = df['Tokens'].apply(lambda x: list(x.sents))
    
    # get number of words
    df['N_words'] = df['Words'].apply(lambda x: len(x))
    
    # get number of sentences
    df['N_sentences'] = df['Sentences'].apply(lambda x: len(x))
    
    return df
  

def total_syllables(df):
    """Get total number of syllables in text."""
    