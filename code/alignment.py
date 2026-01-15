import numpy as np

def ssd_metric(a: np.ndarray, b: np.ndarray) -> float:
    """Sum of Squared Differences — lower is better."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    diff = a - b
    return float(np.sum(diff * diff))

def ncc_metric(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized Cross-Correlation — higher is better."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a_mean = a.mean()
    b_mean = b.mean()
    a_std = a.std() + 1e-8
    b_std = b.std() + 1e-8
    return float(np.sum(((a - a_mean) / a_std) * ((b - b_mean) / b_std)))

def align_bruteforce(
    reference: np.ndarray,
    target: np.ndarray,
    search_range: int = 15,
    metric: str = "ncc",
    edge_crop: float = 0.10,
):
    """
    Exhaustively search integer shifts (dx, dy) in [-search_range, +search_range]
    that best align 'target' to 'reference'. Crops a margin from scoring.
    Returns: best_dx, best_dy, best_score
    """
    assert reference.ndim == 2 and target.ndim == 2, "Input channels must be 2D."
    h, w = reference.shape
    ch = min(max(0, int(h * edge_crop)), h // 4)
    cw = min(max(0, int(w * edge_crop)), w // 4)
    ref_crop = reference[ch:h - ch, cw:w - cw]

    use_ncc = (metric.lower() == "ncc")
    best_score = -np.inf if use_ncc else np.inf
    best_dx = best_dy = 0

    for dy in range(-search_range, search_range + 1):
        for dx in range(-search_range, search_range + 1):
            shifted = np.roll(target, shift=(dy, dx), axis=(0, 1))
            shifted_crop = shifted[ch:h - ch, cw:w - cw]
            if use_ncc:
                score = ncc_metric(ref_crop, shifted_crop)
                if score > best_score:
                    best_score, best_dx, best_dy = score, dx, dy
            else:
                score = ssd_metric(ref_crop, shifted_crop)
                if score < best_score:
                    best_score, best_dx, best_dy = score, dx, dy
    return best_dx, best_dy, float(best_score)

def pyramid_align(
    reference: np.ndarray,
    target: np.ndarray,
    levels: int = 5,
    base_search: int = 4,
    refine_search: int = 2,
    metric: str = "ncc",
    edge_crop: float = 0.10,
):
    """
    Coarse-to-fine multi-scale alignment. Returns (dx_total, dy_total).
    """
    assert reference.ndim == 2 and target.ndim == 2, "Input channels must be 2D."

    refs = [reference]
    tars = [target]
    for _ in range(1, max(1, levels)):
        refs.append(refs[-1][::2, ::2])
        tars.append(tars[-1][::2, ::2])

    dx_total = dy_total = 0
    use_ncc = (metric.lower() == "ncc")

    for lvl in reversed(range(len(refs))):
        ref_l = refs[lvl]
        tar_l = tars[lvl]
        est = np.roll(tar_l, shift=(dy_total, dx_total), axis=(0, 1))

        search = base_search if lvl == 0 else refine_search
        h, w = ref_l.shape
        ch = min(max(0, int(h * edge_crop)), h // 4)
        cw = min(max(0, int(w * edge_crop)), w // 4)
        ref_crop = ref_l[ch:h - ch, cw:w - cw]

        best_dx = best_dy = 0
        best_score = -np.inf if use_ncc else np.inf

        for ddy in range(-search, search + 1):
            for ddx in range(-search, search + 1):
                shifted = np.roll(est, shift=(ddy, ddx), axis=(0, 1))
                shifted_crop = shifted[ch:h - ch, cw:w - cw]
                if use_ncc:
                    score = ncc_metric(ref_crop, shifted_crop)
                    if score > best_score:
                        best_score, best_dx, best_dy = score, ddx, ddy
                else:
                    score = ssd_metric(ref_crop, shifted_crop)
                    if score < best_score:
                        best_score, best_dx, best_dy = score, ddx, ddy

        dx_total += best_dx
        dy_total += best_dy
        if lvl != 0:
            dx_total *= 2
            dy_total *= 2

    return int(dx_total), int(dy_total)
