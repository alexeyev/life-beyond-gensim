# coding: utf-8
import gensim
from gensim.corpora import Dictionary
from gensim.models.hdpmodel import HdpTopicFormatter

from common_reading_utils import get_20ng

# reading data
news_df, word_list_lemmatized = get_20ng()

# preparing data for Gensim
dictionary = Dictionary(word_list_lemmatized)
corpus = [dictionary.doc2bow(text) for text in word_list_lemmatized]

# creating hdp model
print("Training...")
hdp_model = gensim.models.hdpmodel.HdpModel(corpus=corpus, id2word=dictionary)

print(hdp_model.get_topics().shape)

topic_formatter = HdpTopicFormatter(dictionary=dictionary, topic_data=hdp_model.get_topics())

# viewing topics
# Always equals to T, by default 150 :\
# https://medium.com/@diego.garrido.6568/the-implementation-of-hdp-in-gensim-has-a-bug-always-de-number-of-topics-inferred-is-equal-to-t-e86d8da2be5

for topic_id in range(hdp_model.get_topics().shape[0]):
    # best_tokens = [f"{token}:{score:01.4f}" for token, score in topic_formatter.show_topic(topic_id, topn=10)]
    best_tokens = [token for token, score in topic_formatter.show_topic(topic_id, topn=10)]
    print(topic_id, best_tokens)

