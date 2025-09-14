import numpy as np
import craft_text_detector.craft_utils as craft_utils

def safe_adjust_result_coordinates(polys, ratio_w, ratio_h, ratio_net=2):
    """
    Safe replacement for craft_utils.adjustResultCoordinates to avoid
    ValueErrors when polys are of inconsistent shapes.
    """
    valid_polys = [
        p for p in polys
        if isinstance(p, np.ndarray) and p.ndim == 2 and p.shape[1] == 2
    ]

    adjusted_polys = []
    for poly in valid_polys:
        poly = poly * (ratio_w * ratio_net, ratio_h * ratio_net)
        adjusted_polys.append(poly)

    return adjusted_polys

# 🧠 Monkey patch it in memory
craft_utils.adjustResultCoordinates = safe_adjust_result_coordinates
