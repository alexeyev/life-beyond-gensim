# coding: utf-8
import os

import pandas as pd
from sklearn.datasets import fetch_20newsgroups

from preprocess_newsgroups import run_preprocess


def get_20ng():
    news = fetch_20newsgroups(subset='train')

    try:
        news_df: pd.DataFrame = pd.read_csv("20ng_all.csv")
        word_list_lemmatized: pd.DataFrame = pd.read_csv("20ng_word_list.csv")
        word_list_lemmatized = list(line.split() for line in word_list_lemmatized["word_list"].astype(str))
    except FileNotFoundError as e:
        news_df, word_list_lemmatized = run_preprocess(news)
        news_df.to_csv("20ng_all.csv", index=None)
        pd.DataFrame({"word_list": [" ".join(s) for s in word_list_lemmatized]}).to_csv("20ng_word_list.csv",
                                                                                        index=None)

        os.makedirs("octis_data_20ng", exist_ok=True)
        dictionary = list(set([token for text in word_list_lemmatized for token in text]))

        with open("octis_data_20ng/dictionary.txt", "w+", encoding="utf-8") as wf:
            wf.write("\n".join(dictionary))

        pd.DataFrame({"word_list":
                          [" ".join(s) if len(s) > 0 else "emptydocument"
                           for s in word_list_lemmatized]}).to_csv("octis_data_20ng/corpus.tsv",
                                                                   sep="\t", index=None, header=None)

    return news_df, word_list_lemmatized
