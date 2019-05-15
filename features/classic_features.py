"""
Functions for creation of commonly used text readability features.

All functions take a pandas dataframe as an input. The dataframe contains texts in the Text column.
The functions will calculate the feature for every text in the dataframe, creating a column for the feature.

To work, some functions need auxillary features; requirements for each function are written in its description.

List of classic features:
- Avg_words_per_sentence
- Avg_syllables_per_word
- Complex_word_percent
- Difficult_word_percent
- Long_sent_percent
- Long_word_percent
- Avg_letters_per_word
- Comma_percent

List of Auxillary features (features used to calculate other features):
- Tokens
- Words
- Sentences
- N_words
- N_sentences
- N_syllables
- N_polysyllables

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
    Avg_words_per_sentence: average number of words per sentence
    
    Adds auxillary features:
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
    
    # also get average word number per sentence
    df["Avg_words_per_sentence"] = df["N_words"] / df["N_sentences"]
    
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
    Avg_syllables_per_word: average number of syllables per word
    
    Adds auxillary features:
    N_syllables: total number of syllables in the text
    
    :param: the dataframe with the dataset
    :returns: the dataframe with the added features
    """
    
    # get pyphen dictionary
    dic = pyphen.Pyphen(lang='en_EN')
    
    # use pyphen to find the number of hyphens (example: sentence -> sent-ence, 1 hyphen)
    df["N_hyphens"] = df["Text"].apply(lambda x: _count_hyphens(x, dic))
    
    # number of syllables is number of hyphens + number of words 
    # (example: sentence -> sent-ence = 1 hyphen + 1 word = 2 syllables)
    df["N_syllables"] = df["N_words"] + df["N_hyphens"]
    
    # also write average syllable number per word
    df["Avg_syllables_per_word"] = df["N_syllables"] / df["N_words"]
    
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


# POLYSYLLABLES (WORDS WITH 3 OR MORE SYLLABLES)


def _count_polysyllables(words, dic):
    n_complex = 0
    
    for word in words:
        # if the word has more than 3 or more syllables it will have 2 or more hyphens
        if dic.inserted(word).count("-") >= 2:
            n_complex += 1
    
    return n_complex


def polysyllables(df):
    """
    Get total number of polysyllables in text for each text.
    A polysyllable is a word with 3 or more syllables.
    
    Needs features:
    Words
    
    Adds auxillary features:
    N_polysyllables: total number of polysyllables in the text
    
    :param: the dataframe with the dataset
    :returns: the dataframe with the added feature
    """
    
    # get pyphen dictionary
    dic = pyphen.Pyphen(lang='en_EN')
    
    # use pyphen to find the number of polysyllables
    df["N_polysyllables"] = df["Words"].apply(lambda x: _count_polysyllables(x, dic))
    
    return df


# PERCENTAGE OF COMPLEX WORDS (GUNNING FOG)


def complex_words_pct(df):
    """
    Get percentage of complex words as defined by Gunning.
    Complex words (or polysyllables) are those with three or more syllables.
    
    Needs features:
    N_polysyllables
    N_words
    
    Adds features:
    Complex_word_percent: percentage of complex words (Gunning)
    
    :param: the dataframe with the dataset
    :returns: the dataframe with the added feature
    """
     
    # get percentage
    df["Complex_word_percent"] = df["N_polysyllables"] / df["N_words"]
    
    return df


# PERCENTAGE OF LONG SENTENCES (LONGER THAN 25 WORDS)


def _get_n_long_sent(sentences):
    n = 0
    for sentence in sentences:
        if len(sentence) > 25:
            n += 1
    return n


def long_sent_pct(df):
    """
    Get percentage of long sentences.
    Long sentences are defined as having more than 25 words.
    
    Needs features:
    Sentences
    
    Adds features:
    Long_sent_percent: percentage of long sentences
    
    :param: the dataframe with the dataset
    :returns: the dataframe with the added feature
    """
    
    # get percentage
    df["Long_sent_percent"] = df["Sentences"].apply(_get_n_long_sent) / df["N_sentences"]
    
    return df


# PERCENTAGE OF LONG WORDS (LONGER THAN 8 CHARACTERS)


def _get_n_long_word(words):
    n = 0
    for word in words:
        if len(word) > 8:
            n += 1
    return n


def long_word_pct(df):
    """
    Get percentage of long words.
    Long words are defined as having more than 8 chars.
    
    Needs features:
    Words
    
    Adds features:
    Long_word_percent: percentage of long words
    
    :param: the dataframe with the dataset
    :returns: the dataframe with the added feature
    """
    
    # get percentage
    df["Long_word_percent"] = df["Words"].apply(_get_n_long_word) / df["N_words"]
    
    return df


# AVERAGE NUMBER OF LETTERS PER WORD


def _get_n_letters(words):
    n = 0
    for word in words:
        n += len(word)
    return n


def avg_letters_per_word(df):
    """
    Get average number of letters per word.
    
    Needs features:
    Words
    
    Adds features:
    Avg_letters_per_word
    
    :param: the dataframe with the dataset
    :returns: the dataframe with the added feature
    """
    
    # get percentage
    df["Avg_letters_per_word"] = df["Words"].apply(_get_n_letters) / df["N_words"]
    
    return df


# PERCENTAGE OF SENTENCES WITH A COMA


def _get_n_comma_sent(sentences):
    n = 0
    for sentence in sentences:
        if str(sentence).find(",") != -1:
            n += 1
    return n


def comma_pct(df):
    """
    Get percentage of sentences with a comma.
    
    Needs features:
    Sentences
    
    Adds features:
    Comma_percent: percentage of sentences with a comma
    
    :param: the dataframe with the dataset
    :returns: the dataframe with the added feature
    """
    
    # get percentage
    df["Comma_percent"] = df["Sentences"].apply(_get_n_comma_sent) / df["N_sentences"]
    
    return df