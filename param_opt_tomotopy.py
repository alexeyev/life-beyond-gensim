# coding: utf-8

import optuna
import numpy as np
import tomotopy as tp
from common_reading_utils import get_20ng
from common_eval_utils import proportion_unique_words
import matplotlib.pyplot as plt

news_df, word_list_lemmatized = get_20ng()

study = optuna.create_study(direction="maximize", study_name="best_tomotopy_model")

# maybe we should provide a nice starting point
# study.enqueue_trial({"num_topics": 40, "passes": 50, "chunksize": 100})

print("Optuna study created, let's find those hyperparams!")

diversities, coherences = [], []
topn_evaluation = 10


def objective(trial: optuna.trial.Trial):

    # создаём перебираемые гиперпараметры
    num_topics = trial.suggest_int("num_topics", low=15, high=70)
    iter = trial.suggest_int("iter", low=5, high=500, step=5)
    alpha = trial.suggest_float("alpha", low=0.001, high=3, log=True)
    eta = trial.suggest_float("eta", low=0.001, high=3, log=True)

    print(num_topics, iter, alpha, eta)

    mdl = tp.LDAModel(k=num_topics, tw=tp.TermWeight.ONE, alpha=alpha, eta=eta, seed=10)

    for line in word_list_lemmatized:
        mdl.add_doc(line)

    mdl.train(iter=iter, workers=0, parallel=tp.ParallelScheme.DEFAULT)
    print('Log-likelihood: {}'.format(mdl.ll_per_word))
    print("Training done. Evaluation starting.")

    top_terms = []

    for topic_id in range(mdl.k):
        top_terms.append([t for t, s in mdl.get_topic_words(topic_id, top_n=topn_evaluation)])

    coh = tp.coherence.Coherence(mdl, coherence="c_npmi")
    average_coherence = coh.get_score()
    coherence_per_topic = [coh.get_score(topic_id=k) for k in range(mdl.k)]
    std_coherence = np.std(coherence_per_topic)

    # # DIVERSITY: more is better
    diversity = proportion_unique_words(top_terms, topk=topn_evaluation)

    # for topic in top_terms:
    #     print(" ".join(topic))

    print("Diversity:", diversity)
    print(f"Coherence: {average_coherence:.4f}, std: {std_coherence:.3f}")
    print()

    diversities.append(diversity)
    coherences.append(average_coherence)

    # todo: reconsider the formula maybe?
    # diversity \in [0, 1] while npmi \in [-1, 1]
    # 1) diversity MUST be close to 1.0, otherwise the model actually sucks (undertrained)
    # 2) ...but coherence is the most important thing
    # return (diversity if diversity > 0.7 else diversity - 1.0) + 10 * npmi_coherence
    return average_coherence


study.optimize(objective, n_trials=100)

print("The best parameters:")
print("Best score:", study.best_trial.value)
print("Best trial:", study.best_trial.params)

plt.plot(diversities, coherences, "-o")
plt.xlim(0, 1)
plt.ylim(-1, 1)
plt.xlabel("Diversity")
plt.ylabel("Coherence")
plt.show()