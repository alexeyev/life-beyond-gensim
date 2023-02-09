# coding: utf-8
import pandas as pd
from preprocess_newsgroups import run_preprocess
from sklearn.datasets import fetch_20newsgroups  # Read in train subset (11,314 observations)


def get_20ng():

    news = fetch_20newsgroups(subset='train')

    try:
        news_df: pd.DataFrame = pd.read_csv("20ng_all.csv")
        word_list_lemmatized: pd.DataFrame = pd.read_csv("20ng_word_list.csv")
        word_list_lemmatized = list(line.split() for line in word_list_lemmatized["word_list"])
    except FileNotFoundError as e:
        news_df, word_list_lemmatized = run_preprocess(news)
        news_df.to_csv("20ng_all.csv", index=None)
        pd.DataFrame({"word_list": [" ".join(s) for s in word_list_lemmatized]}).to_csv("20ng_word_list.csv",
                                                                                        index=None)
    return news_df, word_list_lemmatized