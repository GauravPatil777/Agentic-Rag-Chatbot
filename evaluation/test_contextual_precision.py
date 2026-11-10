import pytest

from deepeval import assert_test
from deepeval.metrics import ContextualPrecisionMetric

from evaluation.config import judge_model1
from evaluation.evaluation_dataset import test_cases

@pytest.mark.parametrize("test_case", test_cases)
def test_contextual_precision(test_case):

    metric = ContextualPrecisionMetric(
        threshold=0.7,
        model=judge_model1,
        include_reason=True,
        async_mode=False
    )

    assert_test(test_case, [metric])