import pytest

from evaluate import char_accuracy, edit_distance


def test_edit_distance_identical():
    assert edit_distance("1234ABC", "1234ABC") == 0


def test_edit_distance_one_substitution():
    assert edit_distance("1234ABC", "1234ABD") == 1


def test_edit_distance_empty_strings():
    assert edit_distance("", "") == 0
    assert edit_distance("ABC", "") == 3
    assert edit_distance("", "ABC") == 3


def test_char_accuracy_perfect_match():
    assert char_accuracy("1234 ABC", "1234 ABC") == pytest.approx(1.0)


def test_char_accuracy_partial_match():
    acc = char_accuracy("1234ABD", "1234ABC")  # 1 char off out of 7
    assert 0.8 < acc < 1.0


def test_char_accuracy_empty_ground_truth_and_empty_pred():
    assert char_accuracy("", "") == 1.0


def test_char_accuracy_empty_ground_truth_nonempty_pred():
    assert char_accuracy("1234", "") == 0.0
