#!/usr/bin/env python3
""" Ngram BLEU score """
import numpy as np


def create_ngram(sentence, n):
    """ Create new sentence by ngram """
    new_sentence = []
    for i in range(len(sentence) - n + 1):
        n_gram = ""
        for j in range(n):
            n_gram += sentence[i + j]
            if not j + 1 == n:
                n_gram += " "
        new_sentence.append(n_gram)
    return new_sentence


def ngram_bleu(references, sentence, n):
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
    r = np.array([len(ref) for ref in references])[mask].sum()

    cn = create_ngram(sentence, n)
    candidate = {x: 0 for x in cn}
    references = [create_ngram(ref, n) for ref in references]

    max_match = 0
    for ref in references:
        match = 0
        ref_dict = {x: ref.count(x) for x in ref}
        for key in ref_dict.keys():
            if key in candidate:
                match += 1
        if match > max_match:
            max_match = match
    P = max_match / len(cn)
    if c > r:
        BP = 1
    else:
        BP = np.exp(1-(r / c))
    return BP * P
