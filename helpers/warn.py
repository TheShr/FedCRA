import time
import warnings


def warn_segmentation(inside_attribution: float, total_attribution: float) -> None:
    """
    Warn if the inside explanation is greater than total explanation.

    Parameters
    ----------
    inside_attribution: float
        The size of inside attribution.
    total_attribution: float
        The size of total attribution.

    Returns
    -------
    None
    """
    warnings.warn(
        "Inside explanation is greater than total explanation"
        f" ({inside_attribution} > {total_attribution}), returning np.nan."
    )


def warn_empty_segmentation() -> None:
    """
    Warn if the segmentation mask is empty.

    Returns
    -------
    None
    """
    warnings.warn("Return np.nan as result as the segmentation map is empty.")