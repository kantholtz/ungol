# -*- coding: utf-8 -*-

from ungol.retrieval import stats
from ungol.retrieval import clients
from ungol.retrieval import experiment as exp


# helper

def _assert_almost_equal(a: float, b: float):
    assert round(a, 5) == round(b, 5)


#  truth test file
#  - contains empty lines
#  - order is switched
#  - topics have different amounts of docs
#
#  topic1: 3 correct, 2 incorrect
#  topic2: 1 correct, 1 incorrect

truth = exp.Truth.from_file('ungol/tests/test_experiment_truth.txt')


def test_all_topics_there():
    assert 'topic1' in truth
    assert 'topic2' in truth
    assert len(truth) == 2


def test_truth_tasks():
    task = truth['topic1']
    assert task['doc1'] is True
    assert task['doc2'] is False
    assert task['doc3'] is False
    assert task['doc4'] is False
    assert task['doc5'] is True

    task = truth['topic2']
    assert task['doc1'] is False
    assert task['doc2'] is True


def test_truth_correct():
    assert truth['topic1'].correct == 2
    assert truth['topic2'].correct == 1


def test_truth_incorrect():
    assert truth['topic1'].incorrect == 3
    assert truth['topic2'].incorrect == 1


#  test a topic stats instance
#  - use topic1 from the test truth
#  - hitting 1 of 3 correct ones using k=3
#  - one is missing, one is wrong

top_id_t1 = 'topic1'

results_t1 = [
    clients.Result(doc_id='doc1', score=0.6),      # correct
    clients.Result(doc_id='doc2', score=0.5),      # wrong
    clients.Result(doc_id='docnot', score=0.4), ]  # missing

#  Rank    Correct   Precision   Recall    Avg Precision
#     1       True           1      1/2              1/2
#     2      False         1/2      1/2              1/2
#     3      False         1/3      1/2              1/2

# AP: (1/2 * 1 + 0 * 1/2 + 0 * 1/3) = 1/2

stats_t1 = stats.TopicStats.from_search_results(results_t1, truth[top_id_t1])

top_id_t2 = 'topic2'

results_t2 = [
    clients.Result(doc_id='doc1', score=0.9),    # incorrect
    clients.Result(doc_id='doc2', score=0.1), ]  # correct

#  Rank    Correct   Precision   Recall    Avg Precision
#     1      False           0        0               0
#     2       True         1/2        1             1/2

# AP: 0 * 0 + 1 * 1/2 = 1/2

stats_t2 = stats.TopicStats.from_search_results(results_t2, truth[top_id_t2])


def test_stats_k():
    assert stats_t1.k == len(results_t1)
    assert stats_t2.k == len(results_t2)


def test_stats_positives():
    assert stats_t1.positives == truth[top_id_t1].correct
    assert stats_t2.positives == truth[top_id_t2].correct


def test_stats_true_positives():
    assert stats_t1.true_positives == 1
    arr1 = stats_t1.arr_true_positives

    assert len(arr1) == 4
    assert arr1[0] == 0
    assert arr1[1] == 1
    assert arr1[2] == 1
    assert arr1[3] == 1

    assert stats_t2.true_positives == 1
    arr2 = stats_t2.arr_true_positives

    assert len(arr2) == 3
    assert arr2[0] == 0
    assert arr2[1] == 0
    assert arr2[2] == 1


def test_stats_false_positives():
    assert stats_t1.false_positives == 2
    arr1 = stats_t1.arr_false_positives

    assert len(arr1) == 4
    assert arr1[0] == 0
    assert arr1[1] == 0
    assert arr1[2] == 1
    assert arr1[3] == 2

    assert stats_t2.false_positives == 1
    arr2 = stats_t2.arr_false_positives

    assert len(arr2) == 3
    assert arr2[0] == 0
    assert arr2[1] == 1
    assert arr2[2] == 1


def test_stats_false_negatives():
    assert stats_t1.false_negatives == 1
    arr1 = stats_t1.arr_false_negatives

    assert len(arr1) == 4
    assert arr1[0] == 2
    assert arr1[1] == 1
    assert arr1[2] == 1
    assert arr1[3] == 1

    assert stats_t2.false_negatives == 0
    arr2 = stats_t2.arr_false_negatives

    assert len(arr2) == 3
    assert arr2[0] == 1
    assert arr2[1] == 1
    assert arr2[2] == 0


def test_stats_precision():
    _assert_almost_equal(stats_t1.precision, 1/3)
    arr1 = stats_t1.arr_precision

    assert len(arr1) == 4
    _assert_almost_equal(arr1[0], 0)
    _assert_almost_equal(arr1[1], 1)
    _assert_almost_equal(arr1[2], 1/2)
    _assert_almost_equal(arr1[3], 1/3)

    _assert_almost_equal(stats_t2.precision, 1/2)
    arr2 = stats_t2.arr_precision

    assert len(arr2) == 3
    _assert_almost_equal(arr2[0], 0)
    _assert_almost_equal(arr2[1], 0)
    _assert_almost_equal(arr2[2], 1/2)


def test_stats_recall():
    _assert_almost_equal(stats_t1.recall, 1/2)
    arr1 = stats_t1.arr_recall

    assert len(arr1) == 4
    _assert_almost_equal(arr1[0], 0)
    _assert_almost_equal(arr1[1], 1/2)
    _assert_almost_equal(arr1[2], 1/2)
    _assert_almost_equal(arr1[3], 1/2)

    _assert_almost_equal(stats_t2.recall, 1)
    arr2 = stats_t2.arr_recall

    assert len(arr2) == 3
    _assert_almost_equal(arr2[0], 0)
    _assert_almost_equal(arr2[1], 0)
    _assert_almost_equal(arr2[2], 1)


def test_stats_f1():
    _assert_almost_equal(stats_t1.f1, 2/5)
    _assert_almost_equal(stats_t2.f1, 2/3)


def test_stats_avg_precision():
    _assert_almost_equal(stats_t1.avg_precision, 1/2)
    _assert_almost_equal(stats_t2.avg_precision, 1/2)


# test ExperimentStats

top_id_t3 = 'topic3'

stats_t3 = stats.TopicStats(hits=[True, True], positives=2)

exp_stats = stats.ExperimentStats(stats={
    top_id_t2: stats_t2, top_id_t3: stats_t3})


def test_stats_exp_mean_precision():
    # (1/2 + 1) / 2 = 3/4
    _assert_almost_equal(exp_stats.mean_precision, 3/4)


def test_stats_exp_mean_recall():
    # (1 + 1) / 2 = 1
    _assert_almost_equal(exp_stats.mean_recall, 1)


def test_stats_exp_mean_f1():
    # (2/3 + 1) / 2 = 5/6
    _assert_almost_equal(exp_stats.mean_f1, 5/6)


def test_mean_avg_precision():
    # (1/2 + 1) / 2 = 3/4
    _assert_almost_equal(exp_stats.mean_avg_precision, 3/4)
