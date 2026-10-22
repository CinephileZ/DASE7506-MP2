"""Submission entry-point for MP2 candidate selection."""


def predict(bank, arrays, checkpoint):
    """Return one Python integer in 0..3 per question, in bank order.

    Delegates to :func:`predict_method.run` which handles feature extraction,
    model loading, scoring, and tie-breaking.
    """
    return _run_prediction(bank, arrays, checkpoint)


def _run_prediction(bank, arrays, checkpoint):
    from predict_method import run
    return run(bank, arrays, checkpoint)
