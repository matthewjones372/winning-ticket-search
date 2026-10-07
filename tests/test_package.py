import lottery


def test_version_comes_from_package_metadata():
    assert lottery.__version__ == "2.4.0"


def test_public_api_exports_resolve():
    for name in lottery.__all__:
        assert hasattr(lottery, name), name


def test_the_types_needed_to_implement_the_protocols_are_exported():
    for name in [
        "StepCallback",
        "EpochCallback",
        "OptimiserFactory",
        "SchedulerFactory",
        "ParameterSelector",
        "PrunableParameter",
        "ResumableTrainer",
        "RoundRecord",
        "train_with_masks",
    ]:
        assert name in lottery.__all__, name
        assert hasattr(lottery, name), name


def test_round_results_convert_to_and_from_checkpoint_records():
    from lottery import EpochResult, LayerSparsity, Metrics, RoundResult

    result = RoundResult(
        round=1,
        density=0.8,
        epochs=[EpochResult(0, Metrics(1.0, 0.5), Metrics(0.9, 0.6), Metrics(0.8, 0.7))],
        layers=[LayerSparsity("fc.weight", 8, 10)],
        extra_metrics={"quantised": Metrics(0.95, 0.55)},
    )
    assert RoundResult.from_record(result.to_record()) == result
