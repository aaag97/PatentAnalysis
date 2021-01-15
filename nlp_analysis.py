import argparse
import json
import numpy as np
from autocorrect import Speller
spell = Speller()
import spacy
from spacy import displacy
from collections import Counter
import os
nlp = spacy.load("en_core_web_lg")

def get_length(text):
    """
    function to get the length of the claims and add them to a dataframe
    Args:
    text - a text in string form
    Returns:
    row - the length of the text in number of words
    """
    len_ =  len(nlp(text))
    return len_

def get_refs(text):
    """
    function to count all pronouns as well as references to the reader
    Args:
    text - the text in string form
    Returns:
    refs - a dictionnary of words and their corresponding count
    """
    refs = {}
    doc = nlp(text)
    for token in doc:
        if token.pos_ == 'PRON' or token.text.lower() == 'reader':
            word = token.text.lower()
            if not word in refs.keys():
                refs.update({word: 1})
            else:
                refs.update({word: refs[word]+1})
    return refs

def check_neighbors_for_pass(doc,i,nb_neighbors):
    """
    function to check whether there is a auxpass word in the neighbors of a given word
    Args:
    doc - the doc to look in
    i - the index of the central word
    nb_neighbors - the number of neighbors to consider
    Returns:
    a boolean which is True if there is an auxpass word and False otherwise
    """
    if i - nb_neighbors < 0:
        min_ = 0
    else:
        min_ = i-nb_neighbors
    if i + nb_neighbors + 1 >= len(doc):
        max_ = len(doc)
    else:
        max_ = i + nb_neighbors + 1
    for j in range(min_, max_):
        if j != i:
            if doc[j].dep_ == 'auxpass':
                return True
    return False

def get_pass_act_verbs(text, ret_tokens=False):
    """
    function to get passive and active verb forms from a text
    Args:
    text - the text in string form
    ret_tokens - whether to returns strings or token objects
    Returns:
    verbs['active'] - a list of the active verbs
    verbs['passive'] - a list of the passive verbs
    len(verbs['active']) - the number of active verbs
    len(verbs['passive']) - the number of passive verbs
    """
    verbs = {'active':[], 'passive':[]}
    doc = nlp(text)
    verb = ''
    for token in doc:
        if token.dep_ == 'auxpass':
            if ret_tokens: 
                verbs['passive'].append(token)
            else:
                verbs['passive'].append(token.text)
        elif (token.pos_ == 'VERB' and token.dep_ == 'ROOT') or token.pos_ == 'AUX' and token.dep_ != 'ROOT':
            index = token.i
            if not check_neighbors_for_pass(doc,index,2):
                if ret_tokens:
                    verbs['active'].append(token)
                else:
                    verbs['active'].append(token.text)
    return verbs['active'], verbs['passive'], len(verbs['active']), len(verbs['passive'])

#some subtle cleaning
def stick_words(text):
    """
    function to remove the '-' that is often put when words break at the end of a line
    Args:
    text - the text in string form
    Returns:
    the same text with '-' replaced
    """
    return text.replace('- ','')

def get_fin_nonfin_verbs(text, ret_tokens=False):
    """
    function to get finite and non-finite verbs forms from a text
    Args:
    text - the text in string format
    ret_tokens - whether to returns strings or token objects
    Returns:
    verbs['finite'] - a list of the finite form verbs
    len(verbs['finite']) - the number of finite form verbs in the text
    verbs['non-finite'] - a list of the non-finite form verbs
    len(verbs['non-finite']) - the number of non-finite verbs in the text
    """
    verbs = {'finite':[], 'non-finite':[]}
    doc = nlp(text)
    verb = ''
    for token in doc:
        if token.tag_ == 'VB' or token.tag_ == 'VBD' or token.tag_ == 'VBD' or token.tag_ == 'VBG' or token.tag_ == 'VBN' or token.tag_ == 'VBP' or token.tag_ == 'VBZ':
            if token.tag_ == 'VB':
                if ret_tokens:
                    verbs['non-finite'].append(token)
                else:
                    verbs['non-finite'].append(token.text)
            elif token.tag_ == 'VBG':
                if token.i == 0:
                    if ret_tokens:
                        verbs['non-finite'].append(token)
                    else:
                        verbs['non-finite'].append(token.text)
                elif not (doc[token.i-1].tag_ == 'VBD' or doc[token.i-1].tag_ == 'VBP' or doc[token.i-1].tag_ == 'VBP' or doc[token.i-1].tag_ == 'VBZ'):
                    if ret_tokens:
                        verbs['non-finite'].append(token)
                    else:
                        verbs['non-finite'].append(token.text)
            elif token.tag_ == 'VBN':
                if ret_tokens:
                    verbs['non-finite'].append(token)
                else:
                    verbs['non-finite'].append(token.text)
            else:
                if ret_tokens:
                    verbs['finite'].append(token)
                else:
                    verbs['finite'].append(token.text)
    return verbs['finite'], len(verbs['finite']), verbs['non-finite'], len(verbs['non-finite'])


#pronoun count flattening
ref_list = ['i', 'me', 'my', 'you', 'your', 'we', 'reader']
def get_important_pronouns(ref_count):
    """
    function to add the number of each reference to the reader or the author
    Args:
    row - thr row with the column 'Reference Count'
    Returns:
    row - the same row with the following columns - Number of i, Number of me, Number of my, Number of you, Number of your, Number of we, Number of reader
    """
    ref_count_dets = {}
    for noun in ref_list:
        if noun in ref_count.keys():
            ref_count_dets.update({'Number of \'{}\''.format(noun): ref_count[noun]})
        else:
            ref_count_dets.update({'Number of \'{}\''.format(noun): 0})
    return ref_count_dets


def parse_args():
    """
    function to parse command line arguments
    Returns:
    args - an argspace object containing the arguments
    """
    parser=argparse.ArgumentParser(description='Segement and visualize patent pages.')
    parser.add_argument('-i', '--input', type=str, help='the path to the text file', required=True)
    parser.add_argument('-o', '--output', type=str, help='path to the json file to be created', required=True)
    parser.add_argument('-a', '--refs', type=int, help='1 if you would like all refrences to reader and author (not recommended if text contains spelling mistakes as it generates mistakes) and 0 otherwise', required=True)
    args = parser.parse_args()
    return args

def main(input_text, output_path, include_all_refs):
    extracted_info = {}
    len_ = get_length(text=input_text)
    extracted_info.update({'length in words': len_})
    all_refs = get_refs(text=input_text)
    if include_all_refs:
        refs = all_refs
    else:
        refs = get_important_pronouns(ref_count=all_refs)
    extracted_info.update(refs)
    active_list, passive_list, nb_active, nb_passive = get_pass_act_verbs(text=input_text, ret_tokens=False)
    extracted_info.update({'active verbs': active_list, 'passive verbs': passive_list, 'number of active verbs': nb_active, 'number of passive verbs': nb_passive})
    finite_list, finite_len, non_finite_list, non_finite_len = get_fin_nonfin_verbs(text=input_text,ret_tokens=False)
    extracted_info.update({'finite verbs': finite_list, 'non-finite verbs': non_finite_list, 'number of finite verbs': finite_len, 'number of non-finite verbs': non_finite_len})
    with open(output_path, 'w') as f:
        json.dump(extracted_info, f)
    
if __name__ == "__main__":
    args = parse_args()
    input_path = args.input
    output_path = args.output
    include_all_refs = args.refs
    with open(input_path, 'r') as f:
        input_text = f.read()
    main(input_text, output_path, include_all_refs)


