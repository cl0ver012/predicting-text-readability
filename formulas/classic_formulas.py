"""
Commonly used formulas for readability.

All functions take a pandas dataframe as an input. 
To work, required features listed in the description for each formula are needed.
The wanted features can be created using the feature_creation module.

The functions will calculate the formula for every text in the dataframe, creating a column with the result.
"""


def flesch(df):
    """
    Calculates the Flesch formula for each text.
    The formula and its interpretation is given in this wiki page: https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests
    
    Needed features:
    N_words
    N_sentences
    N_syllables
    
    Adds column:
    Flesch - Flesch formula score for the text 
    """
    
    # Flesch formula
    df["Flesch"] = 206.835 - 1.015 * (df["N_words"] / df["N_sentences"]) - 84.6 * (df["N_syllables"] / df["N_words"])
    
    return df


def dale_chall(df):
    """
    Calculates the Dale-Chall formula for each text.
    The formula and its interpretation is given in this wiki page: https://en.wikipedia.org/wiki/Dale%E2%80%93Chall_readability_formula
    
    Needed features:
    N_words
    N_sentences
    Difficult_word_percent
    
    Adds column:
    Dale_Chall - Dale-Chall formula score for the text 
    """

    # Dale-Chall formula
    df["Dale_Chall"] = 0.1579 * (df["Difficult_word_percent"] * 100) + 0.0496 * (df["N_words"] / df["N_sentences"])
    
    return df