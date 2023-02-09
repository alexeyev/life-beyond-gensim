# coding: utf-8

import tomotopy as tp
from common_reading_utils import get_20ng

news_df, word_list_lemmatized = get_20ng()

mdl = tp.HDPModel()

for line in word_list_lemmatized:
    mdl.add_doc(line)

for i in range(0, 100, 10):
    mdl.train(iter=10)
    print('Iteration: {}\tLog-likelihood: {}'.format(i, mdl.ll_per_word))

for k in range(mdl.k):
    if mdl.is_live_topic(k):
        print(k, [t for t, s in mdl.get_topic_words(k, top_n=15)])

# mdl.summary()
