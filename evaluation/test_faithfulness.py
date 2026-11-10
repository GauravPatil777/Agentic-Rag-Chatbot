import pytest

from deepeval import assert_test
from deepeval.metrics import FaithfulnessMetric

from evaluation.config import judge_model2
from evaluation.evaluation_dataset import test_cases

@pytest.mark.parametrize("test_case", test_cases)
def test_faithfulness(test_case):

    metric = FaithfulnessMetric(
        threshold=0.7,
        model=judge_model2,
        include_reason=True,
        async_mode=False
    )

    assert_test(test_case, [metric])