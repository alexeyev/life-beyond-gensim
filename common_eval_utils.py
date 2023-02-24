# coding: utf-8

from typing import List


def proportion_unique_words(list_of_top_terms: List[List[str]], topk=10)-> float:
    """
     Taken (and rewritten) from https://github.com/silviatti/topic-model-diversity/blob/master/diversity_metrics.py
    """
    if topk > len(list_of_top_terms[0]):
        raise Exception('Words in topics are less than ' + str(topk))
    else:
        unique_words = set([term for topic in list_of_top_terms for term in topic[:topk]])
        return len(unique_words) / (topk * len(list_of_top_terms))