#!/usr/bin/env python3
""" Unigram BLEU score """
import numpy as np


def uni_bleu(references, sentence):
    """ calculates the unigram BLEU score for a sentence

        - references is a list of reference translations
            - each reference translation is a list of the
              words in the translation
         -sentence is a list containing the model proposed sentence
        Returns: the unigram BLEU score
    """
    c = len(sentence)
    # Calculate r, closest length reference to candidate
    r_list = np.array([abs(len(ref) - c) for ref in references])
    # Take indices whit the same distance
    mask = np.where(r_list == r_list.min())
    # Apply mask and take the minimum length in masked references
    r = np.array([len(ref) for ref in references])[mask].min()

    candidate = {x: 0 for x in sentence}
    max_match = 0
    for ref in references:
        match = 0
        ref_dict = {x: ref.count(x) for x in ref}
        for key in ref_dict.keys():
            if key in candidate:
                match += 1
        if match > max_match:
            max_match = match
    P = max_match / c

    if c > r:
        BP = 1
    else:
        BP = np.exp(1-(r / c))
    return BP * np.exp(np.log(P).sum() * 1)
