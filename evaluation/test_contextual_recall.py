import pytest

from deepeval import assert_test
from deepeval.metrics import ContextualRecallMetric

from evaluation.config import judge_model2
from evaluation.evaluation_dataset import test_cases

@pytest.mark.parametrize("test_case", test_cases)
def test_contextual_recall(test_case):

    metric = ContextualRecallMetric(
        threshold=0.7,
        model=judge_model2,
        include_reason=True,
        async_mode=False
    )

    assert_test(test_case, [metric])