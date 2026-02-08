# Soft Scattering Guidance: Results

## Setup
- Model: GraphDIT trained on QM9 (discrete categorical diffusion over molecular graphs)
- Target: ethanol (CCO) scattering moments (176-D fingerprint)
- Guidance: inference-time gradient-based steering via differentiable scattering transform
- Applied post-posterior with MOOD-style gradient normalization in logit space
- 10 samples per run, 10 atoms each

## Results

| Configuration | Scale | Start Step | Mean Moment Dist | Min |
|---------------|-------|------------|-----------------|-----|
| No guidance   | 0.0   | —          | **0.91**        | **0.43** |
| Full guidance | 1.0   | 0          | 1.02            | 0.52 |
| Late only     | 1.0   | 250        | 0.95            | 0.45 |
| Late only     | 5.0   | 250        | 1.13            | 0.66 |
| Late only     | 10.0  | 250        | 1.23            | 0.58 |

More guidance = worse. Validity stays 100% in all runs.

## Interpretation

The guidance gradient successfully shifts node (atom type) distributions but has zero effect on edge (bond type) distributions. Edge posteriors are ~98% peaked on "no bond" and cannot be moved in logit space.

Scattering moments encode topology via wavelets on the adjacency matrix. Since edges are immovable, guidance can only relabel atoms on a skeleton fixed by the model. At high scale, this produces nonsensical atom flips (89% N -> 99.99% C) and chemically irrelevant molecules (fluorine-heavy structures for an ethanol target).

## Conclusion

Soft scattering guidance at inference time cannot steer molecular topology because edge distributions in discrete diffusion posteriors are too peaked to perturb. Node-only guidance cannot match a topology-dependent fingerprint.
