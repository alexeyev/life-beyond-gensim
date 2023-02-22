# coding: utf-8

"""
    Sometimes one would like to find out whether
        a) the topics can be mapped onto some known classes/categories,
        b) the model is expressive enough to find clearly different topics
            on some special dataset originally designed for topic classification.

    For that, the model is deliberately trained on the dataset with known labels
    and checked how well the topics 'match' the classes.

    The topic model does not have to match the classes exactly, one can't say this
    is applicable to any topic classification dataset. However, if a clear correspondence
    between the classes and the topics can be seen, this is a good signal, innit?

    Here we implement something like this: https://topicmodels.info/ckling/tmt/part4.pdf
    Slide 21.
"""

import numpy as np
from sklearn.metrics.cluster import homogeneity_completeness_v_measure, mutual_info_score, contingency_matrix
from sklearn.metrics.cluster import rand_score, adjusted_rand_score, adjusted_mutual_info_score


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    c_matrix = contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(c_matrix, axis=0)) / np.sum(c_matrix)


def kl_divergence(actual_distr: np.ndarray, predicted_distr: np.ndarray):
    return np.sum(actual_distr * np.log2(actual_distr / predicted_distr))


def js_distance(array0, array1):
    """ Symmetrical divergence between two distributions """
    return 0.5 * kl_divergence(array0, array1) + 0.5 * kl_divergence(array1, array0)


def jsd_two_sets(set0: np.ndarray, set1: np.ndarray):
    """ Comparing rows
    Dim 0 -- rows count
    Dim 1 -- distribution events
    """
    assert set0.shape[1] == set1.shape[1]

    result = np.zeros((set0.shape[0], set1.shape[0]))

    for i0 in range(set0.shape[0]):
        for i1 in range(set1.shape[0]):
            result[i0, i1] = js_distance(set0[i0].flatten(), set1[i1].flatten())

    return result


def pairwise_score(true_labels, predicted_topics):
    """
        We decide whether the topics are have the same 'class' and compute
        Rand index / Adjusted Rand index / something else
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html

    :param true_labels: a list: class per document
    :param predicted_topics:
    :return:
    """
    pass


def model_greedy_clustering_scores(true_labels, predicted_topics):
    doc2cluster_greedy = np.argmax(predicted_topics, axis=1)
    assert true_labels.shape == doc2cluster_greedy.shape

    homogeneity, completeness, v_measure_score = homogeneity_completeness_v_measure(true_labels, doc2cluster_greedy)

    return {
        "purity": purity_score(true_labels, doc2cluster_greedy),
        "homogeneity": homogeneity,
        "completeness": completeness,
        "v_measure_score": v_measure_score,
        "rand_index": rand_score(true_labels, doc2cluster_greedy),
        "adjusted_rand_index": adjusted_rand_score(true_labels, doc2cluster_greedy),
        "mi": mutual_info_score(true_labels, doc2cluster_greedy),
        "adjusted_mi": adjusted_mutual_info_score(true_labels, doc2cluster_greedy, average_method="arithmetic"),
    }


if __name__ == "__main__":

    import tomotopy as tp
    from common_reading_utils import get_20ng

    news_df, word_list_lemmatized = get_20ng()
    mdl = tp.LDAModel(k=20, seed=10)

    for line in word_list_lemmatized:
        mdl.add_doc(line)

    for i in range(0, 200, 100):
        mdl.train(iter=100)
        print('Iteration: {}\tLog-likelihood: {}\tPerplexity: {}'.format(i, mdl.ll_per_word, mdl.perplexity))

    for k in range(mdl.k):
        print(k, [t for t, s in mdl.get_topic_words(k, top_n=15)])

    doc: tp.Document = None

    doc2topic = np.array([doc.get_topic_dist() for doc in mdl.docs])
    print(model_greedy_clustering_scores(news_df["topic_id"], doc2topic))

    doc2class = np.zeros_like(doc2topic)

    for i, class_id in enumerate(news_df["topic_id"]):
        doc2class[i, class_id] = 1

