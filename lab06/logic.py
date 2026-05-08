import math
from rng import MultiKongGen

def generate_discr_samples(rng: MultiKongGen, values, probs, n: int):
    cum = []
    s = 0.0
    for p in probs:
        s += p
        cum.append(s)
    samples = []
    for _ in range(n):
        u = rng.anabios()
        for i, cp in enumerate(cum):
            if u < cp:
                samples.append(values[i])
                break
    return samples

def chi2_critical_discr(degrees_of_freedom):
    table = {1: 3.841, 2: 5.991, 3: 7.815, 4: 9.488, 5: 11.070, 6: 12.592}
    return table.get(degrees_of_freedom, 9.488)

def compute_discr_stats(samples, values, theor_probs):
    n = len(samples)
    # Теоретические среднее и дисперсия
    theor_mean = sum(v * p for v, p in zip(values, theor_probs))
    theor_var = sum(p * (v - theor_mean)**2 for v, p in zip(values, theor_probs))
    # Эмпирические
    emp_mean = sum(samples) / n
    emp_var = sum((x - emp_mean)**2 for x in samples) / n
    # Частоты и эмпирические вероятности
    freq = [0] * len(values)
    for x in samples:
        idx = values.index(x)
        freq[idx] += 1
    emp_probs = [f / n for f in freq]
    # χ²
    chi2 = 0.0
    for i, f_obs in enumerate(freq):
        f_exp = n * theor_probs[i]
        if f_exp > 0:
            chi2 += (f_obs - f_exp)**2 / f_exp
    # Критическое значение и гипотеза
    df = len(values) - 1
    chi2_crit = chi2_critical_discr(df)
    hypothesis = "Принимается" if chi2 <= chi2_crit else "Отвергается"
    chi2_comparison = f"{chi2:.2f} {'≤' if chi2 <= chi2_crit else '>'} {chi2_crit}"
    
    # Относительные погрешности
    mean_err = abs(emp_mean - theor_mean) / theor_mean * 100 if theor_mean != 0 else 0
    var_err = abs(emp_var - theor_var) / theor_var * 100 if theor_var != 0 else 0
    return {
        "n": n,
        "theor_mean": theor_mean,
        "theor_var": theor_var,
        "emp_mean": emp_mean,
        "emp_var": emp_var,
        "mean_error": mean_err,
        "var_error": var_err,
        "chi2": chi2,
        "chi2_crit": chi2_crit,
        "hypothesis": hypothesis,
        "chi2_comparison": chi2_comparison,
        "freq": freq,
        "emp_probs": emp_probs
    }

def generate_norm_samples(rng: MultiKongGen, n: int):
    samples = []
    for _ in range(n):
        u_sum = sum(rng.anabios() for _ in range(12))
        samples.append(u_sum - 6.0)
    return samples

NORM_BINS = [
    (-float('inf'), -2.0), (-2.0, -1.0), (-1.0, 0.0),
    (0.0, 1.0), (1.0, 2.0), (2.0, float('inf'))
]
NORM_BIN_LABELS = ["(-∞, -2)", "[-2, -1)", "[-1, 0)", "[0, 1)", "[1, 2)", "[2, +∞)"]

def norm_cdf(x):
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

NORM_THEO_PROBS = [norm_cdf(hi) - norm_cdf(lo) for lo, hi in NORM_BINS]
NORM_CHI2_CRIT = 11.07

def compute_norm_stats(samples):
    n = len(samples)
    emp_mean = sum(samples) / n
    emp_var = sum((x - emp_mean)**2 for x in samples) / n
    freq = [0] * len(NORM_BINS)
    for x in samples:
        for i, (lo, hi) in enumerate(NORM_BINS):
            if (lo == -float('inf') or x >= lo) and (hi == float('inf') or x < hi):
                freq[i] += 1
                break
    chi2 = 0.0
    for i, f_obs in enumerate(freq):
        f_exp = n * NORM_THEO_PROBS[i]
        if f_exp > 0:
            chi2 += (f_obs - f_exp)**2 / f_exp
    mean_abs_err = abs(emp_mean - 0.0)
    mean_rel_err = mean_abs_err * 100 if abs(0.0) > 1e-9 else 0
    var_rel_err = abs(emp_var - 1.0) / 1.0 * 100
    return {
        "n": n, "emp_mean": emp_mean, "emp_var": emp_var,
        "mean_abs_err": mean_abs_err, "mean_rel_err": mean_rel_err,
        "var_rel_err": var_rel_err, "chi2": chi2,
        "chi2_crit": NORM_CHI2_CRIT, "freq": freq,
        "theo_probs": NORM_THEO_PROBS, "bin_labels": NORM_BIN_LABELS
    }