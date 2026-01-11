# Raman spectrum analysis utilities

`raman.py` runs the full analysis pipeline on raw Spectrum Studio data.

Some other files are also runnable as standalone programs, either exposing a CLI to their functionality or running demos/tests.

We're currently experimenting with the pipeline, and not everything implemented here is used. Collectively, these modules support:
- Polynomial fitting, used for fluorescent baseline reduction
- Savitzky-Golay smoothing
- Peak detection and feature extraction via Hilbert Vibration Decomposition
    - Zhao, X.Y.; Liu, G.Y.; Sui, Y.T.; Xu, M.; Tong, L. Denoising method for Raman spectra with low signal-to-noise ratio based on feature extraction. *Spectrochimica Acta Part A* 2020, *250*, 119374. DOI: [10.1016/j.saa.2020.119374](https://doi.org/10.1016/j.saa.2020.119374)
- Peak detection
