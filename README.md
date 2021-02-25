# Microhalo models
Tools for predicting the formation and evolution of dark matter halos, particularly at the smallest scales.

## halo_formation.py

Samples a halo population using the method of [arXiv:1905.05766](https://arxiv.org/abs/1905.05766) (modified based on Appendix A of [arXiv:1910.08553](https://arxiv.org/abs/1910.08553)). Density peaks are sampled from a matter power spectrum, and the structure of each peak predicts the structure of the resulting halo.

## tidal_evolution.py

Predicts the evolution of a subhalo due to a host halo's tidal forces using the model described in [arXiv:1906.10690](https://arxiv.org/abs/1906.10690) (with an addition based on Appendix C of [arXiv:1910.08553](https://arxiv.org/abs/1910.08553)).

## stellar_encounters.py

Predicts the evolution of a halo due to an arbitrary sequence of encounters with point objects, such as stars. The model is described in [arXiv:1907.13133](https://arxiv.org/abs/1907.13133)

# Examples

To be added.

# Planned additions

1. A predictive model of the outcomes of mergers between microhalos.
2. For certain power spectra, smooth accretion is rapid enough to drive internal halo evolution. Predictively model this effect.
3. A more general model of subhalo tidal evolution.
