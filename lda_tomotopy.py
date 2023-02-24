# coding: utf-8
import tomotopy as tp

from common_reading_utils import get_20ng

news_df, word_list_lemmatized = get_20ng()

print(word_list_lemmatized[0][:7])

mdl = tp.LDAModel(k=20)

for line in word_list_lemmatized:
    mdl.add_doc(line)

for i in range(0, 100, 10):
    mdl.train(iter=10)
    print('Iteration: {}\tLog-likelihood: {}'.format(i, mdl.ll_per_word))

for k in range(mdl.k):
    print(k, [t for t, s in mdl.get_topic_words(k, top_n=15)])

# calculate coherence using preset
for preset in ('u_mass', 'c_uci', 'c_npmi', 'c_v'):
    coh = tp.coherence.Coherence(mdl, coherence=preset)
    average_coherence = coh.get_score()
    coherence_per_topic = [coh.get_score(topic_id=k) for k in range(mdl.k)]
    print('==== Coherence : {} ===='.format(preset))
    print('Average:', average_coherence, '\nPer Topic:', coherence_per_topic)
    print()
