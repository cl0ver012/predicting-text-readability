"""
Functions for creation of commonly used text readability features.

All functions take a pandas dataframe as an input. The dataframe contains texts in the Text column.
The functions will calculate the feature for every text in the dataframe, creating a column for the feature.
"""
import pandas as pd

# for tokenization
import spacy

# for finding number of syllables
import pyphen

# the following spacy model has to be downloaded
SPACY_MODEL = "en_core_web_sm"


# WORDS AND SENTENCES


def _get_words(x):
    words = [token.text for token in x if token.is_punct != True]
    return words


def words_and_sentences(df):
    """
    Uses spacy to find number of words and sentences for each text.
    
    Adds features:
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
    df['Words'] = df['Tokens'].apply(_get_words)
    
    # get sentences
    df['Sentences'] = df['Tokens'].apply(lambda x: list(x.sents))
    
    # get number of words
    df['N_words'] = df['Words'].apply(lambda x: len(x))
    
    # get number of sentences
    df['N_sentences'] = df['Sentences'].apply(lambda x: len(x))
    
    return df


# SYLLABLES   


def _count_hyphens(text, dic):
    return dic.inserted(text).count("-")


def syllables(df):
    """
    Get total number of syllables in text for each text.
    
    Needs features:
    N_words
    
    Adds features:
    N_syllables: total number of syllables in the text
    
    :param: the dataframe with the dataset
    :returns: the dataframe with the added feature
    """
    
    # get pyphen dictionary
    dic = pyphen.Pyphen(lang='en_EN')
    
    # use pyphen to find the number of hyphens (example: sentence -> sent-ence, 1 hyphen)
    df["N_hyphens"] = df["Text"].apply(lambda x: _count_hyphens(x, dic))
    
    # number of syllables is number of hyphens + number of words 
    # (example: sentence -> sent-ence = 1 hyphen + 1 word = 2 syllables)
    df["N_syllables"] = df["N_words"] + df["N_hyphens"]
    
    # we don't need the number of hyphens anymore
    df.drop(columns=["N_hyphens"], inplace=True)
    
    return df


# PERCENTAGE OF DIFFICULT WORDS (DALE-CHALL)


def _get_dale_chall_easy_words():
    easy_words = set()
    
    with open("resources/dale_chall_easy_word_list.txt") as file:
        lines = [line.rstrip('\n') for line in file]
        
        for line in lines:
            easy_words.add(line.lower())
    
    return easy_words


def _get_num_difficult_words(text, easy_words):
    n = 0
    for word in text:
        if word.lower() not in easy_words:
            n += 1
    return n


def difficult_words_pct(df):
    """
    Get percentage of difficult words as required for Dale-Chall formula. 
    Word is counted as difficult if it's not in Dale-Chall easy word list.
    
    Needs features:
    Words
    N_words
    
    Adds features:
    Difficult_word_percent - percentage of difficult words (Dale-Chall)
    
    :param: the dataframe with the dataset
    :returns: the dataframe with the added feature
    """
    
    easy_words = _get_dale_chall_easy_words()
    
    df["Difficult_word_percent"] = df["Words"].apply(lambda x: _get_num_difficult_words(x, easy_words)) / df["N_words"]
    
    return df


# PERCENTAGE OF COMPLEX WORDS (GUNNING FOG)


def _count_complex_words(words, dic):
    n_complex = 0
    
    for word in words:
        # if the word has more than 3 or more syllables it will have 2 or more hyphens
        if dic.inserted(word).count("-") >= 2:
            n_complex += 1
    
    return n_complex


def complex_words_pct(df):
    """
    Get percentage of complex words as defined by Gunning.
    Complex words are those with three or more syllables.
    
    Needs features:
    Words
    N_words
    
    Adds features:
    Complex_word_percent: percentage of complex words (Gunning)
    
    :param: the dataframe with the dataset
    :returns: the dataframe with the added feature
    """
    
    # get pyphen dictionary
    dic = pyphen.Pyphen(lang='en_EN')
    
    # use pyphen to find the number of complex words
    df["N_complex_words"] = df["Words"].apply(lambda x: _count_complex_words(x, dic))
    
    # get percentage
    df["Complex_word_percent"] = df["N_complex_words"] / df["N_words"]
    
    # we don't need the number of complex words anymore
    df.drop(columns=["N_complex_words"], inplace=True)
    
    return df