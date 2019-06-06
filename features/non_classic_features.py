"""
Functions for creation non-classic text readability features.
Non-classic features are features which use more complex NLP techniques like parse trees.

All functions take a pandas dataframe as an input. The dataframe contains texts in the Text column.
The functions will calculate the features for every text in the dataframe, creating a column each feature.

To work, some functions need auxillary features; requirements for each function are written in its description.

List of non-classic features:
- NP_per_sent
- VP_per_sent
- PP_per_sent
- SBAR_per_sent
- SBARQ_per_sent
- avg_NP_size
- avg_VP_size
- avg_PP_size
- avg_parse_tree

List of Auxillary features (features used to calculate other features):
- Tokens

"""
from collections import Counter, defaultdict
import pandas as pd
import spacy

# benepar dependency
from benepar.spacy_plugin import BeneparComponent

# the following spacy model has to be downloaded
SPACY_MODEL = "en_core_web_sm"

# the following benepar model has to be downloaded
BENEPAR_MODEL = "benepar_en_small"


# PARSE-TREE FEATURES


def _parse_tree_height(sent):
    """
    Gets the height of the parse tree for a sentence.
    """
    children = list(sent._.children)
    if not children:
        return 0
    else:
        return max(_parse_tree_height(child) for child in children) + 1


def _get_constituents(tokens):
    """
    Gets the number and avg. length of each constituent. 
    """
    
    const_counter = Counter()
    const_lengths = defaultdict(list)

    for sentence in tokens.sents:
        for const in sentence._.constituents:
            # add constituent to constituent counter
            const_counter.update(Counter(const._.labels))
            
            # append the length of the constituent
            for label in const._.labels:
                const_lengths[label].append(len(const))
    
    # for each constituent, get average of constituent's lengths
    const_avgs = defaultdict(int)
    for key in const_lengths.keys():
        avg = 0.0
        for length in const_lengths[key]: 
            avg += length
        avg /= len(const_lengths[key])
        
        const_avgs[key] = avg
         
    return const_counter, const_avgs


def _get_parse_tree_height(tokens):
    """
    Get average parse tree height of each sentence.
    """
    avg_parse_tree_height = 0.0
    
    for sentence in tokens.sents:
        avg_parse_tree_height += _parse_tree_height(sentence)
        
    n_sentences = len(list(tokens.sents))
    avg_parse_tree_height /= n_sentences
    
    return avg_parse_tree_height, n_sentences


def _get_parse_tree_features(tokens):
    const_counter, const_avgs = _get_constituents(tokens)
    avg_parse_tree_height, n_sentences = _get_parse_tree_height(tokens)
    
    NP_per_sent = const_counter['NP'] / n_sentences
    VP_per_sent = const_counter['VP'] / n_sentences
    PP_per_sent = const_counter['PP'] / n_sentences
    SBAR_per_sent = const_counter['SBAR'] / n_sentences
    SBARQ_per_sent = const_counter['SBARQ'] / n_sentences
    avg_NP_size = const_avgs['NP']
    avg_VP_size = const_avgs['VP']
    avg_PP_size = const_avgs['PP']
    avg_parse_tree = avg_parse_tree_height
    
    return NP_per_sent, VP_per_sent, PP_per_sent, \
        SBAR_per_sent, SBARQ_per_sent, avg_NP_size, \
        avg_VP_size, avg_PP_size, avg_parse_tree
    

def parse_tree_features(df):
    """
    Get features which can be extracted from the parse tree of a text. 
    
    Adds features:
    NP_per_sent: NPs (noun phrase) / num of sentences
    VP_per_sent: VPs (verb phrase) / num of sentences
    PP_per_sent: PPs (prepositional phrase) / num of sentences
    SBAR_per_sent: SBARs (subordinate clause) / num of sentences
    SBARQ_per_sent: SBARQs (direct question introduced by wh-element) / num of sentences
    avg_NP_size: Average lenght of an NP
    avg_VP_size: Average lenght of an VP
    avg_PP_size: Average lenght of an PP
    avg_parse_tree: Average height of a parse Tree
    
    :param: the dataframe with the dataset
    :returns: the dataframe with the added features
    """
    
    nlp = spacy.load(SPACY_MODEL, disable=['ner'])
    nlp.add_pipe(BeneparComponent("benepar_en_small"))
    
    # parse text
    df['B_Tokens'] = df['Text'].apply(lambda x: nlp(x))
    
    # get features
    df['NP_per_sent'], df['VP_per_sent'], df['PP_per_sent'], \
    df['SBAR_per_sent'], df['SBARQ_per_sent'], df['avg_NP_size'], \
    df['avg_VP_size'], df['avg_PP_size'], df['avg_parse_tree'] = zip(*df['B_Tokens'].map(_get_parse_tree_features))
    
    # remove B_Tokens
    df.drop(columns=["B_Tokens"], inplace=True)
    
    return df

    

