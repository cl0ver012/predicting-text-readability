"""
Commonly used formulas for readability.

All functions take a pandas dataframe as an input. 
To work, required features listed in the description for each formula are needed.
The wanted features can be created using the feature_creation module.

Formulas implemented:
- Flesch formula
- Dale-Chall formula
- Gunning fog index

The functions will calculate the formula for every text in the dataframe, creating a column with the result.
"""


# FLESCH 


def flesch(df):
    """
    Calculates the Flesch formula for each text.
    The formula and its interpretation is given in this wiki page: https://en.wikipedia.org/wiki/Flesch%E2%80%93Kincaid_readability_tests
    
    Needed features:
    Avg_words_per_sentence
    Avg_syllables_per_word
    
    Adds column:
    Flesch - Flesch formula score for the text 
    """
    
    # Flesch formula
    df["Flesch"] = 206.835 - 1.015 * df["Avg_words_per_sentence"] - 84.6 * df["Avg_syllables_per_word"]
    
    return df


# DALE-CHALL


def dale_chall(df):
    """
    Calculates the Dale-Chall formula for each text.
    The formula and its interpretation is given in this wiki page: https://en.wikipedia.org/wiki/Dale%E2%80%93Chall_readability_formula
    
    Needed features:
    Avg_words_per_sentence
    Difficult_word_percent
    
    Adds column:
    Dale_Chall - Dale-Chall formula score for the text 
    """

    # Dale-Chall formula
    df["Dale_Chall"] = 0.1579 * (df["Difficult_word_percent"] * 100) + 0.0496 * df["Avg_words_per_sentence"]
    
    # adjust if percentage of difficul words is greater than 5%
    df.loc[df["Difficult_word_percent"] > 0.05, "Dale_Chall"] += 3.6365
        
    return df


# GUNNING FOG


def gunning_fog(df):
    """
    Calculates the Gunning fog formula for each text.
    The formula and its interpretation is given in this wiki page: https://en.wikipedia.org/wiki/Gunning_fog_index
    
    Needed features:
    Avg_words_per_sentence
    Complex_words_percent
    
    Adds column:
    Gunning_fog - Gunning fog score for the text 
    """

    # Gunning fog formula
    df["Gunning_fog"] = 0.4 * (df["Avg_words_per_sentence"] + 100 * df["Complex_word_percent"])
    
    return df