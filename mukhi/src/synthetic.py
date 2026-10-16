"""Synthetic activations with planted structure. Used by tests and by the
pipeline smoke-check so the whole chain can be exercised without model access."""
import numpy as np


def build_synthetic(n_items=300, n_layers=12, dim=64, seed=0,
                    converges=True, noise=1.0):
    """
    Returns (acts_gurmukhi, acts_shahmukhi, acts_urdu), each
    (n_layers, n_items, dim).

    converges=True : script fades with depth, content and language grow.
    converges=False: nothing changes with depth (correct null). NOTE it does
        NOT hold w_script fixed while growing w_sem -- that lowers probe
        accuracy without removing script information and is an artefact, not
        a null. See results/FINDING_02.
    """
    rng = np.random.default_rng(seed)
    semantic = rng.normal(size=(n_items, dim))
    urdu_sem = rng.normal(size=(n_items, dim))

    script_dir = rng.normal(size=dim); script_dir /= np.linalg.norm(script_dir)
    lang_dir = rng.normal(size=dim); lang_dir /= np.linalg.norm(lang_dir)

    ag = np.zeros((n_layers, n_items, dim))
    ash = np.zeros((n_layers, n_items, dim))
    au = np.zeros((n_layers, n_items, dim))

    for L in range(n_layers):
        t = L / (n_layers - 1)
        if converges:
            w_script, w_sem, w_lang = 6.0 * (1 - t), 6.0 * t, 6.0 * t
        else:
            w_script, w_sem, w_lang = 6.0, 3.0, 0.0

        ag[L] = w_sem*semantic + (-w_script)*script_dir + (-w_lang)*lang_dir \
                + noise*rng.normal(size=(n_items, dim))
        ash[L] = w_sem*semantic + (w_script)*script_dir + (-w_lang)*lang_dir \
                 + noise*rng.normal(size=(n_items, dim))
        au[L] = w_sem*urdu_sem + (w_script)*script_dir + (w_lang)*lang_dir \
                + noise*rng.normal(size=(n_items, dim))
    return ag, ash, au
