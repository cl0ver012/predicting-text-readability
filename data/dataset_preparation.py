"""
Dataset preparation and cleaning script
 
This script converts the dataset from the given format 
(a lot of text files in 5 folders belonging to 5 classes)
into a CSV format such that every row has a cleaned text
and the class. The dataset is also divided into a train
and a test set.

The following things are done:
1) Converting into a pandas dataframe
2) Dataset cleaning
3) Examples from class 4 are undersampled
4) Split into train and test
5) Saving data in CSV format
"""
import pandas as pd
import os

# for language detection
from langdetect import detect_langs

# for train-test split
from sklearn.model_selection import train_test_split


# CONVERTING INTO A PANDAS DATAFRAME


def get_weebit_as_dataframe():
    """
    Gets the WeeBit dataset which is stored in many text files as a single dataframe.
    The dataframe has two columns - text and the readability level of the text.
    
    :returns: WeeBit dataset as a pandas dataframe
    """

    # there are 5 levels of readability in the WeeBit dataset
    levels = [0, 1, 2, 3, 4]
    dataset_path = "./WeeBit/"

    texts = list()
    for level in levels:
        files = os.listdir(dataset_path + str(level))
        for file in files:
            with open(dataset_path + str(level) + "/" + file, 'r', encoding='latin-1') as txt_file:
                # read the entire text as string (texts are quite small)
                text_string = txt_file.read()
                texts.append([text_string, level])
            
    # create Pandas dataframe from all texts            
    df = pd.DataFrame(texts, columns = ['Text', 'Level'])
    
    return df


# DATASET CLEANING


def _get_english_prob(langs):
    return {result.lang: result.prob for result in langs}.get('en', 0.0)


def _remove_non_english(df, english_prob_threshold = 0.99):
    """
    Helper function which removes all texts for which there is a significant (default: >=1%) probability of being non-English.
    """
    langs = df['Text'].apply(detect_langs)
    english_probs = langs.apply(_get_english_prob)
    df = df[english_probs > english_prob_threshold]
    return df


# lines which are not related to the text (copyright, Flash warnings, and similar)
NON_CONTEXT_LINES = ['This page is best viewed in an up-to-date web browser with style sheets (CSS) enabled.',
                     'While you will be able to view the content of this page in your current browser, you will not be able to get the full visual experience.',
                     'Please consider upgrading your browser software or enabling style sheets (CSS) if you are able to do so.',
                     'The BBC is not responsible for the content of external internet sites.',
                     'For information on how to enable JavaScript please go to the',
                     'You will not be able to see this content until you have JavaScript switched on.',
                     'Your web browser does not have JavaScript switched on at the moment.',
                     'You have disabled Javascript, or are not running Javascript on this browser.',
                     'Go to the',
                     'go to the',
                     'The enhanced version of the site requires the Flash 8 plugin (or higher) to be installed and JavaScript to be enabled on your browser.',
                     'To find out how to turn on JavaScript',
                     'The enhanced version of the site requires the Flash 8 plugin (or higher) to be installed and JavaScript to be enabled on your browser.',
                     'To find out how to install a Flash plugin,',
                     'The enhanced version of the site requires the Flash 8 plugin (or higher) to be installed and JavaScript to be enabled on your browser.',
                     'Download the Adobe Flash player to view this conten.',
                     'All trademarks and logos are property of Weekly Reader Corporation.',
                     'measures published under license with MetaMetrics, Inc.']


def _remove_non_content_lines(text):
    """
    Helper function for removing non-context lines defined in NON_CONTEXT_LINES constant
    """
    for line in NON_CONTEXT_LINES:
        text = text.replace(line, '')
    return text.strip()


def clean_weebit(df):
    """
    Cleans the WeeBit dataset. 
    1) All newlines in text are replaced by spaces.
    2) Empty texts are removed.
    3) Duplicate texts are removed.
    4) Non-English texts are removed.
    5) Non-content lines are removed.
    
    :param df: dataframe of the WeeBit dataset
    :returns: Cleaned dataframe
    """
    
    # convert to string, remove None values
    df['Text'] = df['Text'].astype(str)
    df.dropna(inplace=True)
    
    # replace all newlines with spaces
    df['Text'] = df['Text'].str.replace(".\n", ". ")
    df['Text'] = df['Text'].str.replace("\n", ". ")
    
    # remove empty texts
    df = df[df['Text'].str.len() != 0]
    
    # remove all duplicates
    df = df.drop_duplicates("Text")
    
    # remove now english texts
    df = _remove_non_english(df)
    
    # remove non-content lines
    df['Text'] = df['Text'].apply(_remove_non_content_lines)
    
    df['Text'] = df['Text'].astype(str)
    df.reset_index(drop=True, inplace=True)
    return df


# CLASS 4 UNDERSAMPLING


def level_4_undersampling(df, n_level4 = 800):
    """
    Undersamples the examples belonging to level 4 readability level.
    There is around 10x more examples for this level. 
    To prevent class imbalance, only a number of examples from level 4 class will be used. 
    
    :param df: WeeBit dataset dataframe
    :param n_level4: the number of sampled examples for level 4 class
    :returns: WeeBit dataset dataframe with undersampled level 4 class
    """
    
    df_4 = df[df.Level == 4]
    df_ = df[df.Level != 4]
    
    df_4 = df_4.sample(n=n_level4)
    
    df = pd.concat([df_, df_4])
    
    df.reset_index(drop=True, inplace=True)
    return df


# MAIN 


TEST_SIZE = 0.2
DATASET_CSV = "weebit.csv"
TRAIN_SET_CSV = "weebit_train.csv"
TEST_SET_CSV = "weebit_test.csv"


def main():
    """
    Main script function. Does the following:
    1) Read the WeeBit dataset from files
    2) Cleans the dataset
    3) Undersamples level 4 class
    4) Splits dataset into train and test sets
    5) Saves train, test and whole dataset into CSV files.
    The file names are defined by TRAIN_SET_CSV, TEST_SET_CSV and DATASET_CSV constants.
    """
    
    # get WeeBit dataframe
    df = get_weebit_as_dataframe()
    print("Read WeeBit as dataframe.")
    
    # clean dataset
    df = clean_weebit(df)
    print("Cleaned the dataset.")
    
    # undersample level 4 class
    df = level_4_undersampling(df)
    print("Undersampled level 4 class.")
    
    # split into train and test set
    train_df, test_df = train_test_split(df,
                                     test_size = TEST_SIZE,
                                     shuffle = True,
                                     stratify = df.Level)
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)
    print("Split into train and test set.")
    
    # write into csv
    df.to_csv(DATASET_CSV, encoding='utf-8')
    train_df.to_csv(TRAIN_SET_CSV, encoding='utf-8')
    test_df.to_csv(TEST_SET_CSV, encoding='utf-8')
    print("Saved to csv.")
    print("Final dataset is:")
    print(df.Level.value_counts())
    

if __name__ == "__main__":
    main()