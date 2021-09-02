# Microhalo models
Tools for predicting the formation and evolution of dark matter halos, particularly at the smallest scales.

Requires: numpy and scipy.

## halo_formation.py

Samples a halo population using the method of [arXiv:1905.05766](https://arxiv.org/abs/1905.05766) (modified based on Appendix A of [arXiv:1910.08553](https://arxiv.org/abs/1910.08553)). Density peaks are sampled from a matter power spectrum, and the structure of each peak predicts the structure of the resulting halo.

**These predictions are very accurate when the matter power spectrum exhibits a steep boost at some small scale** (for example, due to early matter domination, axion PQ symmetry breaking, or particle production during inflation, among other cosmological scenarios). **They fare less well with flatter power spectra**, like standard CDM, for which (a) halo mergers are much more important and (b) smooth accretion may be rapid enough to drive internal halo evolution.

## tidal_evolution.py

Predicts the evolution of a subhalo due to a host halo's tidal forces using the model described in [arXiv:1906.10690](https://arxiv.org/abs/1906.10690) (with an addition based on Appendix C of [arXiv:1910.08553](https://arxiv.org/abs/1910.08553)).

In order to model the range of scales relevant to microhalo scenarios, these tidal evolution predictions have been validated across a much broader range of subhalo-host systems and timescales than are ordinarily seen in cosmological simulations. The J factor evolution is the most tightly tuned, whereas there is more scatter from the r_max and v_max predictions.

## stellar_encounters.py

Predicts the evolution of a halo due to an arbitrary sequence of encounters with point objects, such as stars. The model is described in [arXiv:1907.13133](https://arxiv.org/abs/1907.13133) (with a refinement to the postencounter halo density profile given in Appendix B of [arXiv:2109.xxxxx](https://arxiv.org/abs/2109.xxxxx)).

# Examples

1. [extragalactic-annihilation](examples/extragalactic-annihilation): Computes the factor by which microhalos arising due to early matter domination boost the extragalactic annihilation signal.
2. [annihilation-suppression](examples/annihilation-suppression): Integrates the aggregate factor by which tidal evolution scales the annihilation rate within subhalos of a set internal density inside a set host halo.
3. [pulsar-timing](examples/pulsar-timing): Generates a sample of halos in the solar vicinity for the purpose of pulsar timing observations.

# Planned additions

1. A predictive model of the outcomes of mergers between microhalos.
2. For certain power spectra, smooth accretion may be rapid enough to drive internal halo evolution. Explore this effect.
3. A more general model of subhalo tidal evolution would be nice.
