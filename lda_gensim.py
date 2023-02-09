# coding: utf-8

import gensim
from gensim.corpora import Dictionary

from common_reading_utils import get_20ng

# reading data
news_df, word_list_lemmatized = get_20ng()

# preparing data for Gensim
dictionary = Dictionary(word_list_lemmatized)
corpus = [dictionary.doc2bow(text) for text in word_list_lemmatized]

# creating lda model
print("Training...")
lda_model = gensim.models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=20)


# viewing topics
for topic_id in range(lda_model.get_topics().shape[0]):
    # best_tokens = [f"{token}:{score:01.4f}" for token, score in topic_formatter.show_topic(topic_id, topn=10)]
    best_tokens = [token for token, score in lda_model.show_topic(topic_id, topn=10)]
    print(topic_id, best_tokens)

