# Travelling-Wave Detection and Analysis Methods for `cogpy.travelling_waves`

## Main survey

This survey focuses on **methods** (not dashboards) for answering “is there a travelling wave, what direction/speed/frequency/wavelength/morphology, how transient, how confident?” in **spatiotemporal neural recordings** (ECoG grids, MEAs, depth/laminar arrays with geometry, widefield voltage/calcium imaging, EEG/MEG topographies when treated as space–time fields). It draws heavily on wave-specific neuroscience methods (e.g., **phase-gradient and pattern-field approaches**) and on mature adjacent-field methods (e.g., **array processing / f–k beamforming** and **multidimensional multitaper spectral estimation**) when they transfer cleanly. citeturn8search11turn13search5turn9search1turn10view1

### Taxonomy of travelling-wave analysis methods

Below is a method taxonomy organized by **what mathematical object is estimated** and **what assumptions are required**. Wherever possible, the entries list: input representation → estimated quantities → failure modes → practicality under noise, small grids, irregular layouts, and nonstationarity.

#### Spectral and array-processing methods

**Conceptual idea.** Travelling waves correspond to concentrated energy along a **dispersion relation** between temporal frequency and spatial wavenumber. In the simplest plane-wave case, a dominant wavevector **k** and temporal frequency ω satisfy ω ≈ v·k (or |v| ≈ ω/|k|), so energy clusters at particular (k_x, k_y, ω) or along ridges in (k, ω). In array-processing form, this becomes estimation of **slowness** (1/velocity) and propagation direction from multi-sensor phase delays. citeturn9search1turn4search8turn23academia40

**Core variants.**
- **3D FFT / k–ω spectrum (regular grids).** Operates on a regularly sampled cube x×y×t (or 1D space×t). Output is a 3D spectrum magnitude |F(k_x,k_y,ω)|; peaks/ridges provide direction and speed estimates (plus dominant spatial wavelength λ=2π/|k|). Works best when spatial aperture is large enough to resolve k, and when waves are sufficiently stationary in the analysis window. (This is common in wave physics and signal processing; in neuroscience it appears most naturally in imaging and high-density arrays, and is also used conceptually in “standing vs travelling” decompositions.) citeturn23academia40turn5search6turn9search1  
- **f–k / beamforming scans (irregular or arbitrary sensor geometries).** Compute cross-spectral structure across sensors at frequency f, then scan candidate slowness vectors and compute beam power (“f–k spectrum”). The “high-resolution” MVDR/Capon form adaptively shapes the spatial filter and is foundational in array processing for resolving multiple directions. citeturn9search1turn9search4turn4search8  
- **Capon / MVDR and related high-resolution methods.** Widely used in array seismology/infrasound/sensor arrays to estimate propagation direction and slowness from coherent structure. In wave-detection contexts it provides a principled way to separate multiple simultaneous plane waves (to the extent the array aperture and SNR allow). citeturn9search1turn4search35

**Mathematical object.** Multidimensional Fourier transform or cross-spectral matrices across sensors; maxima in k-space / slowness-space. citeturn9search1turn4search8

**Required inputs.**
- Regular-grid k–ω: data on a grid with known spacing and time sampling.
- Beamforming f–k: time series + sensor coordinates (can be irregular). citeturn4search8turn23academia40

**Outputs.**
- Direction (angle of k or backazimuth), speed (via ω/|k|, or slowness), wavelength λ, sometimes multiple simultaneous waves if multiple peaks exist. citeturn4search8turn9search1

**Assumptions.**
- Dominant plane-wave component within each window/frequency (or a small mixture).
- Approximate stationarity within the window.
- For f–k: wavefront approximately planar over the array aperture (far-field assumption in classical array processing). citeturn4search8turn9search1

**Strengths.**
- Strong physical interpretability; pairs naturally with **synthetic validation** (inject known plane waves and recover k, ω).
- Beamforming handles **irregular sensor layouts** and is a direct bridge from mature geophysics/proximity sensing literature to neural arrays. citeturn4search8turn4search35

**Failure modes.**
- **Small spatial grids**: poor k-resolution and leakage; ambiguity between standing vs travelling components when aperture is small.
- **Nonstationarity**: time-varying direction/speed smears spectral peaks unless windowing or adaptive time-frequency representations are used.
- **Multiple wave components**: peaks overlap, producing biased k estimates unless multi-component methods (multi-peak detection, MUSIC/ESPRIT-like, or MVDR) are used. citeturn9search1turn4search35

**Computational cost.**
- 3D FFT: O(N log N) per window (fast, GPU-friendly).
- Beamforming scan: O(N_sensors^2 × N_grid) per frequency/time window if done naïvely; can be manageable for ECoG/MEA sizes with careful vectorization/caching. citeturn9search1turn4search8

**Nonstationarity support.**
- Requires windowed methods (STFT-in-time) or time-frequency representations; can be extended with wavelets or multitaper time-frequency. citeturn4search29turn7search2

##### Special attention A: 3D / spatiotemporal spectral analysis and multidimensional multitaper

**What multidimensional multitaper is.** Multitaper (Thomson) reduces spectral variance and leakage by averaging spectra computed under multiple orthogonal tapers (DPSS/Slepian sequences). citeturn9search13turn7search2

**3D multitaper feasibility.**
- A simple, explicit and implementation-friendly multidimensional generalization constructs multidimensional tapers as **outer products of 1D tapers**, producing separable 2D/3D tapers. This is precisely described by entity["people","Alfred Hanssen","signal processing author"] (1997), including extension to higher dimensions. DOI: 10.1016/S0165-1684(97)00076-5. citeturn10view1  
- In practice, neuroscience Python ecosystems commonly provide **1D multitaper** (time only) but not turnkey 3D multitaper; nonetheless, separable outer-product tapers make a clean first implementation target because it leverages existing DPSS generators (SciPy) and standard FFTs. citeturn10view1turn7search22turn7search2

**Is multidimensional multitaper used in practice?** It is mature in signal processing and appears in the multidimensional random-field literature (and is standard conceptually in array processing), but it is not yet a mainstream “push-button” tool in neural wave analysis pipelines compared with phase-gradient methods. citeturn10view1turn9search13turn13search5

**Suitability for small (x,y) grids with long time axis.**
- If x,y are small (e.g., 8×8 or 10×10), spatial wavenumber resolution is intrinsically limited, so the primary value of 3D multitaper becomes **variance reduction and leakage control**, not magical spatial super-resolution.
- A practical compromise is: multitaper along **time** (many samples → many useful tapers), modest tapering along **space** (few tapers), then estimate k-structure either via FFT-in-space (regular grids) or beamforming (irregular geometries). This design follows directly from the separable-taper framework. citeturn10view1turn7search22turn7search2

#### Phase-based travelling-wave detection and characterization

**Conceptual idea.** Many neural travelling waves are expressed as **spatial phase gradients** in band-limited oscillations. If an oscillation’s instantaneous phase φ(x,y,t) changes smoothly across space at a given time, then the gradient ∇φ points along the direction of phase increase; combined with instantaneous frequency ω(t)=∂φ/∂t, one obtains a local **phase velocity** estimate v ≈ ω · (∇φ / |∇φ|^2) (up to conventions), and direction is given by ∇φ. citeturn3search2turn9search22turn7search9

**Canonical examples in neuroscience.**
- entity["people","Doug Rubino","neuroscience author"] et al. (2006) used multielectrode motor cortex recordings and showed propagating waves mediating information transfer (Nat Neurosci), and their methodology is frequently treated as foundational for phase-gradient extraction in 2D arrays. DOI: 10.1038/nn1802. citeturn2search0turn2search16  
- entity["people","Honghui Zhang","neuroscience author"] et al. (2018) provided a widely adopted ECoG-grid implementation: Hilbert phase on band-limited alpha/theta oscillations; fit a plane to phase; derive direction/speed; introduce phase-gradient directionality (PGD) as robustness. DOI: 10.1016/j.neuron.2018.05.019. citeturn9search22turn3search2  
- entity["people","Evgueniy V Lubenov","neuroscience author"] & entity["people","Athanassios G Siapas","neuroscience author"] (2009) is canonical for travellers along a 1D axis (hippocampal theta), highlighting how phase gradients and travelling waves are naturally linked. DOI: 10.1038/nature08010. citeturn9search3  
- entity["people","Sayak Bhattacharya","neuroscience author"] et al. (2022) demonstrate that waves are often **rotating** rather than purely planar in microelectrode PFC arrays, motivating methods that go beyond global plane fits. DOI: 10.1371/journal.pcbi.1009827. citeturn20view0turn18view0

**Mathematical object.** Analytic signal a(t)=x(t)+i·H[x(t)] (Hilbert) or a generalized analytic-signal representation; instantaneous phase φ and its spatial gradient ∇φ; circular statistics on phase residuals. citeturn3search2turn24view0turn5search5

**Required inputs.**
- Multi-channel signals with sensor positions (grid or irregular).
- Typically requires bandpass filtering (or a broadband phase method) to make phase meaningful. citeturn9search22turn24view0turn5search5

**Outputs.**
- Direction, speed, local wavelength, phase coherence/robustness, plus (with additional steps) classification into planar vs rotating patterns. citeturn3search2turn20view0turn13search5

**Assumptions.**
- Oscillatory structure where instantaneous phase is well-defined.
- Smoothness in space (at least locally) and sufficiently dense sampling to estimate gradients. citeturn3search2turn5search5

**Strengths.**
- Highly interpretable for oscillatory neurophysiology; produces direct, time-resolved direction and speed estimates at the timescale of cycles, and it is widely supported by canonical electrophysiology travelling-wave papers. citeturn2search0turn9search22turn20view0turn9search3

**Failure modes.**
- **Filter dependence**: narrowband filtering can distort waveforms and can create apparent phase structure if applied incautiously; broadband signals can break analytic-signal assumptions (negative-frequency issues) unless specialized representations are used. citeturn24view0turn5search5  
- **Phase wraps / unwrapping**: spatial phase unwrapping is nontrivial in small or noisy arrays and near singularities; wrong unwrapping biases gradient estimates. citeturn2search0turn3search2  
- **Standing waves and sequential activations** can mimic travelling waves at the sensor level, particularly in extracranial EEG/MEG; contemporary critiques emphasize this ambiguity, especially for large-scale sensor-space claims. citeturn8search12turn7search11turn2search28

**Small grids and irregular layouts.**
- On small grids, local finite-difference gradients are unstable; global regression (plane-wave fit) is preferred.
- Irregular layouts need gradient estimation via local regression, Delaunay/graph gradients, or model-based fitting (see Plane-wave / delay-surface models below). citeturn23search5turn11search29

##### Special attention D: phase-based wave detection details and robustness

**Key implementation patterns seen in the literature.**
- The phase-gradient/plane-fit approach from entity["people","Honghui Zhang","neuroscience author"] et al. (2018) explicitly converts phase maps into direction and speed (via plane-fit parameters) and uses PGD as a robustness/“how planar is the phase map?” metric. citeturn3search2turn9search22  
- entity["people","Zachary W Davis","neuroscience author"] et al. (2020) introduced “generalized phase” to stabilize instantaneous phase estimation for wideband signals without relying solely on narrowband filtering; their public MATLAB code documents the motivation (centering analytic signal, correcting negative-frequency components). DOI: 10.1038/s41586-020-2802-y. citeturn5search21turn24view0turn5search5

**Practical robustness guidance derived from published methods.**
- Treat phase-based estimates as meaningful primarily where local oscillatory amplitude/coherence is sufficient; otherwise down-weight or mask.
- Use explicit goodness-of-fit measures (PGD-like, circular variance, residual dispersion) as *first-class outputs*, not an afterthought. This is directly consistent with PGD practice and with toolboxes that test against surrogates. citeturn3search2turn24view2turn13search5

#### Plane-wave and delay-surface fitting

**Conceptual idea.** Travelling waves imply systematic inter-sensor delays. Estimate those delays and fit an explicit **propagation model**:
- Plane-wave model: τ_i ≈ (k·r_i)/ω + b, or equivalently phase φ_i ≈ k·r_i + b (mod 2π).
- Delay-surface model: τ(x,y) solves a fitted surface; its gradient gives slowness/direction.

This family overlaps strongly with phase-gradient methods but expands to **non-oscillatory events** by using lags/cross-correlation rather than phase.

**Canonical neuroscience usage.**
- Early human EEG travelling-wave work includes regression of delays/slopes across sensors (planar wave direction via slope sign), e.g., entity["people","T M Patten","neuroscience author"] et al. (2012) in PLOS ONE (Human Cortical Traveling Waves). DOI: 10.1371/journal.pone.0038392. citeturn11search33  
- Recent protocols emphasize explicit regression/statistical testing for travelling waves in microelectrode arrays, e.g., entity["people","V M Zarr","neuroscience author"] et al. (2025) in STAR Protocols (multi-linear regression approach, event-based). citeturn7search7turn8search3

**Mathematical object.** Regression on phases or delays; robust estimators (e.g., circular–linear regression, least squares on delay surfaces, robust regression on lags).

**Required inputs.**
- Sensor coordinates + either phase estimates (oscillatory waves) or lag estimates (cross-correlation/event timing). citeturn11search33turn7search7

**Outputs.**
- Direction and speed (or slowness), model fit residuals/uncertainty. citeturn9search22turn4search8

**Assumptions.**
- Within the fitted window/event, propagation is approximately coherent and described by a low-parameter model (plane or simple radial model). citeturn4search8turn11search33

**Strengths.**
- Works on **small grids** because it uses global constraints rather than local derivatives.
- Works on **irregular layouts** naturally (regress on actual coordinates). citeturn4search8turn23search5

**Failure modes.**
- Multiple simultaneous waves or curved/spiral waves can produce poor plane fits; residuals are informative but classification requires additional machinery. citeturn13search5turn14view0

##### Special attention E: when explicit plane/delay models work and fail

**Where they work best.**
- Arrays with coherent, near-planar propagation during a time window (common in many ECoG alpha/theta travelling waves reported in invasive recordings). citeturn2search0turn9search22turn11search33  
- Event-like propagation where delays are meaningful even without stable oscillatory phase (as in explicit regression protocols). citeturn7search7turn8search3

**Where they fail.**
- Rotating/spiral/source–sink patterns: a single wavevector cannot represent the field; methods that infer velocity vector fields and/or phase singularities are needed. citeturn20view0turn14view0turn17view0

#### Motion-estimation methods, including optical flow

**Conceptual idea.** Treat spatiotemporal neural activity patterns as a “movie” and estimate a dense velocity field describing how patterns move between frames. This is widely developed in computer vision and has been explicitly ported to neural-wave analysis both in imaging and in multielectrode recordings. citeturn12view0turn13search5turn13search3

**Key neuroscience-specific toolchains.**
- entity["people","Navvab Afrashteh","neuroimaging author"] et al. (2017) created a MATLAB optical-flow toolbox (OFAMM/OFAMM-like), comparing Horn–Schunck, combined local–global, and temporospatial methods on simulated and mouse voltage/calcium imaging data, concluding combined local–global performed best for wave dynamics in their tests. DOI: 10.1016/j.neuroimage.2017.03.034. citeturn12view0turn13search15turn24view3  
- entity["people","Rory G Townsend","computational neuroscience author"] & entity["people","Pulin Gong","computational neuroscience author"] (2018) link travelling waves to coherent-structure ideas (vortices) and introduce velocity vector fields and pattern classification; their MATLAB toolbox (NeuroPattToolbox) includes surrogate/noise-driven checks. DOI: 10.1371/journal.pcbi.1006643. citeturn13search5turn24view2turn13search5  
- entity["people","L Cao","neuroscience author"] et al. (2021) explicitly describe using **optical flow** to characterize propagating spatiotemporal LFP patterns in hippocampal-array recordings (Cell Reports Methods). citeturn3search17turn6search17

**Mathematical object.** A velocity field u(x,y,t) satisfying an optical-flow constraint (brightness/feature constancy + regularization), often solved with variational methods. citeturn13search13turn13search32turn13search3

**Required inputs.**
- A 2D lattice of frames, e.g., amplitude maps, phase maps, analytic-signal real/imag maps, or imaging frames (ΔF/F).
- For irregular arrays, a preprocessing step to interpolate to a grid (with careful caveats) or an adaptation of flow to scattered data. citeturn13search5turn23search5

**Outputs.**
- Dense or semi-dense velocity vectors; derived divergence/curl; critical points (sources/sinks/vortices/spirals); wave trajectories. citeturn24view2turn12view0

**Assumptions.**
- Small inter-frame displacements, some form of constancy, and spatial smoothness regularization (classic constraints). citeturn13search13turn13search32turn13search3

**Strengths.**
- Naturally handles **complex morphologies** (rotations, spirals) when paired with vector-field topology (divergence/curl/critical points), matching the “spiral/source/sink” vocabulary common in modern wave-pattern papers and toolboxes. citeturn13search5turn24view2turn14view0turn17view0

**Failure modes.**
- Optical flow estimates **apparent motion** and can be biased by amplitude modulations or interference patterns; oscillatory phase wrapping can destabilize flow unless the representation is chosen carefully (e.g., compute flow on unwrapped phase or on complex analytic-signal components). The existence of multiple overlapping waves can yield flows that are not physically meaningful as a single propagation velocity. citeturn13search13turn24view0turn13search5

**Computational cost.**
- Variational optical flow is moderate but tractable for typical grid sizes; mature implementations exist (e.g., scikit-image provides iterative Lucas–Kanade pyramidal flow and TV-L1 flow). citeturn13search3turn13search6

##### Special attention B: optical flow for oscillatory neural fields—best practices and limitations

**Evidence of use in neural waves.**
- Explicit optical-flow toolboxes exist for mesoscale imaging and have been evaluated with simulated ground truth and real voltage/calcium imaging (Afrashteh et al., 2017). citeturn12view0turn24view3  
- Optical flow is also integrated into pattern-classification frameworks for neural recordings (Townsend & Gong, 2018; NeuroPattToolbox). citeturn13search5turn24view2  
- Optical flow has been used on array-recorded LFP-derived spatiotemporal features (Cao et al., 2021). citeturn6search17turn3search17

**Practical recommendations for `cogpy` (method-driven, not UI-driven).**
- Prefer computing flow on **band-limited analytic-signal derived maps** (amplitude, phase, or complex components) rather than raw broadband signals, unless a broadband phase representation is used (e.g., generalized phase) to stabilize phase. citeturn12view0turn24view0turn5search5  
- Treat optical flow as an **estimator of local phase-velocity structure** rather than a definitive “true axonal propagation” measurement; pair it with independent validation metrics (e.g., plane-fit residuals, k–ω spectral peaks) to avoid overinterpretation. This caution aligns with ongoing debates about sensor-level wave interpretations. citeturn7search11turn8search12turn13search5

#### Decomposition methods: complex PCA/SVD, DMD, tensor factorization

**Conceptual idea.** Travelling waves often manifest as **low-dimensional spatiotemporal structure**. Decompositions aim to recover modes that encode travelling/standing components, possibly with rotation and propagation.

**Transferable methods with concrete code.**
- **Complex PCA (CPCA).** By encoding signals as complex (e.g., analytic signals or phase maps), CPCA can separate standing and travelling structure. This approach appears directly in brain-wide spatiotemporal pattern work and comes with public code repositories. citeturn5search2turn5search30turn5search6  
- **Dynamic Mode Decomposition (DMD).** Originating in fluid mechanics, DMD approximates a linear operator whose eigenvectors/modes carry oscillatory dynamics; it is intended to recover coherent spatiotemporal structures and has extensive theoretical grounding and references. citeturn5search39turn5search7turn5search3  
- **Rotation-focused linear methods (e.g., jPCA interpretations).** Some recent work argues that neuronal “rotational dynamics” can be explained by travelling waves, reinforcing the need for tools that connect rotations in low-dimensional projections back to propagating patterns. citeturn6search1turn7search5

**Mathematical object.** Linear operators/modes in complex or real space; eigen-decompositions; low-rank approximations.

**Strengths.**
- Useful for summarizing dynamics and separating components.
- Can complement direct wave estimators by providing alternative evidence for coherent travelling structure. citeturn5search2turn5search7

**Failure modes.**
- Decompositions can produce “wave-like” modes from smoothness and shifts, including in situations where the physical interpretation is subtle; caution is warranted, and null/surrogate testing should be standard. citeturn5search14turn7search11

#### Bayesian, state-space, and switching-state methods

**Conceptual idea.** Represent travelling-wave parameters (direction, speed, wave type) as latent variables that evolve over time, possibly switching among discrete regimes (e.g., two dominant directions). Use probabilistic filtering/smoothing to estimate time-varying wave state and uncertainty.

**What exists in neuroscience.**
- Switching state-space models are explicitly motivated for neural time series with rapid changes in dynamics, but they are not (yet) common as standardized travelling-wave detectors; they are better viewed as a “Phase 2+” architecture target. citeturn6search4  
- Many travelling-wave papers implicitly treat wave direction as state-dependent and bidirectional (e.g., task modulation), suggesting a natural role for switching models. citeturn20view0turn23search7turn11search30

**What is mature in adjacent fields.**
- Particle filtering and tracking approaches are used for wavefront propagation in excitable media (e.g., cardiac wavefront tracking demonstrations), indicating algorithmic feasibility for wavefront/state tracking when a measurement model is specified. citeturn6search2turn5search4

**Mathematical object.** Latent wave parameters with transition dynamics; observation models mapping latent wave state to sensor measurements; posterior distributions.

**Strengths.**
- Natural uncertainty quantification; handles nonstationarity by design.

**Failure modes.**
- Requires a committed generative model and careful identifiability work; heavy implementation burden compared to regression/spectral/phase-gradient methods.

##### Special attention C: Kalman/state-space methods for travelling waves—what’s realistic for `cogpy`

**Current state of the field.** The most direct evidence base for Kalman filtering in travelling-wave *parameter tracking* is stronger in signal processing and wavefront tracking domains than in mainstream neuroscience travelling-wave toolkits. citeturn6search2turn6search4

**A pragmatic `cogpy` framing.**
- Treat state-space tracking as an **optional wrapper** around per-window wave estimates (from PGD/plane-fit, k–ω peaks, or optical flow): the measurement is (k_t, ω_t) or (direction_t, speed_t) with uncertainty; a Kalman filter smooths time variation and can support detecting state changes. This is a conceptually clean extension of widely used per-window approaches and aligns with general switching-state models for neural dynamics. citeturn6search4turn3search2turn13search5

#### Methods for complex wave morphologies: spirals, sources/sinks, multiple waves

**Conceptual idea.** Spirals and rotating waves are organized around **phase singularities** (points where the phase is undefined and rotation occurs). Sources/sinks correspond to divergence structure in velocity fields; spirals correspond to nonzero curl and topological charge.

**Neuroscience evidence base (spirals are not niche).**
- fMRI “brain spirals” were analyzed in entity["people","Yiben Xu","computational neuroscience author"] et al. (2023), explicitly describing spiral-like rotational wave patterns organized around phase singularity centers. DOI: 10.1038/s41562-023-01626-5. citeturn14view0turn14view0  
- Sleep spindles forming travelling spiral waves were reported in entity["people","Yiben Xu","computational neuroscience author"] et al. (2025, Communications Biology), also emphasizing phase singularities and rotational patterns. DOI: 10.1038/s42003-025-08447-4. citeturn17view0turn15view0  
- Toolboxes like NeuroPatt explicitly include critical point analysis and vector-field decompositions intended to classify patterns beyond planar waves. citeturn24view2turn13search5

**Algorithmic building block (transferable).** Phase singularity detection via winding number / topological charge estimators is mature in excitable media analysis (especially cardiac mapping), with explicit formulae and comparative evaluations. citeturn5search4turn5search12turn5search28

##### Special attention F: classification and detection of spirals/rotations/radial waves

**Good candidates for `cogpy` that are methodologically explicit.**
- **Phase singularity via local winding number** in a complex field (analytic signal) on a grid: robust, interpretable, and well aligned with both neuroscience spiral-wave papers and excitable-media methodology. citeturn5search12turn14view0turn17view0  
- **Velocity-field topology**: compute divergence/curl of velocity fields (from optical flow) and classify sources/sinks/vortices, consistent with Townsend & Gong style. citeturn13search5turn24view2

**Challenges.**
- Needs careful handling of amplitude nulls, phase noise, and spatial interpolation if sensors are not on a true grid. citeturn23search5turn5search12

### Foundational and modern papers by method family

The list below intentionally prioritizes papers with implementable detail and direct relevance to invasive electrophysiology/imaging arrays, while including adjacent-field algorithm sources where they are canonical and transferable.

**Phase-gradient / plane-fit travelling waves (core electrophysiology).**
- entity["people","Doug Rubino","neuroscience author"] et al. *Propagating waves mediate information transfer in the motor cortex.* **Nature Neuroscience** (2006). DOI: 10.1038/nn1802. Domain: motor cortex electrophysiology (2D array). Estimates: propagating waves; phase gradients/delays. Why it matters: canonical early implementation for 2D multielectrode travelling wave extraction; underlies many later phase-gradient pipelines. citeturn2search0turn2search16  
- entity["people","Honghui Zhang","neuroscience author"] et al. *Theta and Alpha Oscillations Are Traveling Waves in the Human Neocortex.* **Neuron** (2018). DOI: 10.1016/j.neuron.2018.05.019. Domain: human ECoG. Estimates: direction & speed from fitted phase gradient; PGD robustness metric. Code: method detail in open article and widely reproduced; this is the most direct “Phase 1” blueprint for `cogpy` plane-wave fitting. citeturn9search22turn3search2  
- entity["people","Evgueniy V Lubenov","neuroscience author"] & entity["people","Athanassios G Siapas","neuroscience author"]. *Hippocampal theta oscillations are travelling waves.* **Nature** (2009). DOI: 10.1038/nature08010. Domain: rodent hippocampus (1D axis). Estimates: phase gradients and travelling direction. Why it matters: canonical 1D travelling-wave example; useful for validating 1D implementations. citeturn9search3turn9search6  
- entity["people","Sayak Bhattacharya","neuroscience author"] et al. *Traveling waves in the prefrontal cortex during working memory.* **PLOS Computational Biology** (2022). DOI: 10.1371/journal.pcbi.1009827. Domain: microelectrode arrays. Estimates: planar vs rotating waves; wave direction trends. Code: public MATLAB repository for analysis. Why it matters: motivates non-planar classification and provides reusable code patterns. citeturn20view0turn24view1  
- entity["people","T M Patten","neuroscience author"] et al. *Human Cortical Traveling Waves: Dynamical Properties and Correlates.* **PLOS ONE** (2012). DOI: 10.1371/journal.pone.0038392. Domain: human EEG. Estimates: planar wave direction and slope-based measures. Why it matters: illustrates plane-wave regression logic in extracranial settings and highlights sensor-space ambiguity issues. citeturn11search33  

**Broadband phase and phase robustness.**
- entity["people","Zachary W Davis","neuroscience author"] et al. *Spontaneous travelling cortical waves gate perception in behaving primates.* **Nature** (2020). DOI: 10.1038/s41586-020-2802-y. Domain: primate cortex electrophysiology. Estimates: travelling waves and behavioural relevance; introduces generalized-phase handling for wideband signals. Code: public “generalized-phase” repository describing algorithmic corrections. Why it matters: provides a method to reduce the brittleness of narrowband Hilbert-phase pipelines. citeturn5search21turn24view0turn5search5  

**Optical flow / velocity-field pattern analysis.**
- entity["people","Navvab Afrashteh","neuroimaging author"] et al. *Optical-flow analysis toolbox for characterization of spatiotemporal dynamics in mesoscale optical imaging of brain activity.* **NeuroImage** (2017). DOI: 10.1016/j.neuroimage.2017.03.034. Domain: mouse voltage/calcium imaging. Estimates: velocity fields, sources/sinks, trajectories; compares Horn–Schunck/CLG/temporospatial. Code: public repo and MATLAB distribution. Why it matters: tested toolbox with simulation + experimental data; directly reusable method patterns for imaging-like arrays. citeturn12view0turn24view3turn13search1  
- entity["people","Rory G Townsend","computational neuroscience author"] & entity["people","Pulin Gong","computational neuroscience author"]. *Detection and analysis of spatiotemporal patterns in brain activity.* **PLOS Computational Biology** (2018). DOI: 10.1371/journal.pcbi.1006643. Domain: neural population recordings. Estimates: multiple wave classes; velocity fields; coherent-structure framing. Code: NeuroPattToolbox (MATLAB) with surrogate testing and pattern transitions. Why it matters: clearest implementable blueprint for detecting spirals/sources/sinks as well as planar waves. citeturn13search5turn24view2  
- entity["people","L Cao","neuroscience author"] et al. *Uncovering spatial representations from spatiotemporal decoding of hippocampal field potentials.* **Cell Reports Methods** (2021). Domain: hippocampal array LFP features. Includes optical flow for propagating pattern characterization. Why it matters: directly bridges optical flow from imaging into electrophysiology array feature maps. citeturn3search17turn6search17turn13search9  

**Spiral/rotational wave morphology and phase singularities.**
- entity["people","Yiben Xu","computational neuroscience author"] et al. *Interacting spiral wave patterns underlie complex brain dynamics and are related to cognitive processing.* **Nature Human Behaviour** (2023). DOI: 10.1038/s41562-023-01626-5. Domain: fMRI. Estimates: spiral waves, phase singularity centers, task relevance. Why it matters: motivates phase-singularity detection and multi-spiral interactions in `cogpy` (even if modality differs). citeturn14view0turn5search16  
- entity["people","Yiben Xu","computational neuroscience author"] et al. *Spatiotemporal dynamics of sleep spindles form spiral waves…* **Communications Biology** (2025). DOI: 10.1038/s42003-025-08447-4. Domain: high-density EEG. Estimates: spiral dynamics, phase singularities, trajectories. Why it matters: shows spiral-wave patterning at scale and strongly motivates robust spiral metrics and trajectory statistics. citeturn17view0turn15view0  
- entity["people","H Lilienkamp","physics author"] et al. *Detecting spiral wave tips using deep learning.* **Scientific Reports** (2021). DOI: 10.1038/s41598-021-99069-3. Domain: excitable media. Why it matters: not recommended as a first method for `cogpy`, but the paper succinctly documents classical phase-singularity definitions used as ground truth and highlights performance issues near noise. citeturn5search12  

**Spectral / k–ω / array-processing foundations transferable to neural arrays.**
- entity["people","J Capon","signal processing author"]. *High-resolution frequency-wavenumber spectrum analysis.* **Proceedings of the IEEE** (1969). DOI: 10.1109/PROC.1969.7278. Domain: array processing. Estimates: high-resolution f–k spectrum (MVDR/Capon). Why it matters: direct foundation for f–k beamforming modules that work on irregular or small neural arrays with appropriate assumptions. citeturn9search1turn9search4  
- entity["people","D J Thomson","signal processing author"]. *Spectrum estimation and harmonic analysis.* **Proceedings of the IEEE** (1982). DOI: 10.1109/PROC.1982.12433. Domain: multitaper spectral estimation. Why it matters: foundation for multitaper time-frequency estimation used pervasively; provides principled variance reduction and (with jackknife variants) uncertainty machinery that can be carried into wave metrics. citeturn9search13turn9search5  
- entity["people","Alfred Hanssen","signal processing author"]. *Multidimensional multitaper spectral estimation.* **Signal Processing** (1997). DOI: 10.1016/S0165-1684(97)00076-5. Domain: multidimensional spectral estimation. Why it matters: clear recipe for separable 2D/3D DPSS tapers for 3D k–ω estimation and for robust windowed spectral maps. citeturn10view1  
- entity["people","Brett M Wingeier","neurophysics author"] et al. *Spherical harmonic decomposition applied to spatial-temporal analysis of human high-density EEG.* (2000, arXiv). Domain: spatial spectral analysis on hemispherical/irregular sampling. Why it matters: suggests a practical route for spatial spectra on non-grid sensor layouts (spherical harmonics) and quantifies sampling requirements using simulations. citeturn23academia40  

**Nonstationary large-scale phase dynamics and irregular sampling (relevant to iEEG/SEEG).**
- entity["people","David M Alexander","neuroscience author"] et al. *Large-scale cortical travelling waves predict localized future cortical signals.* **PLOS Computational Biology** (2019). DOI: 10.1371/journal.pcbi.1007316. Domain: ECoG + MEG. Why it matters: demonstrates travelling-wave-like large-scale eigenvectors from Fourier/PCA features, reinforcing decomposition + spectral framings. citeturn23search3turn23search30  
- entity["people","David M Alexander","neuroscience author"] & entity["people","Laura Dugué","neuroscience author"]. *The dominance of large-scale phase dynamics in human cortex, from delta to gamma.* (bioRxiv 2024/2026 versions). Domain: irregularly sampled iEEG spatial spectra estimation. Why it matters: points directly at the “irregular geometry spatial spectrum” problem `cogpy` will face for SEEG and sparse grids. citeturn23search4turn23search5  

**Directionality measures adjacent-but-useful (validation/contrast).**
- entity["people","G Nolte","neuroscience author"] et al. *Robustly Estimating the Flow Direction of Information in Complex Physical Systems.* (2008) introduces the phase-slope index (PSI) and its robustness properties against mixing, which can be used as an auxiliary directionality metric distinct from travelling-wave kinematics. FieldTrip documents sign interpretation for PSI. citeturn11search6turn11search36  

### Code ecosystem survey

This section is intentionally **wave-method-centric**: only codebases that implement travelling-wave detection/analysis logic (or directly implement enabling blocks such as optical flow solvers) are cataloged.

image_group{"layout":"carousel","aspect_ratio":"16:9","query":["phase gradient traveling waves ECoG grid","k omega spectrum frequency wavenumber analysis example","optical flow vector field brain imaging traveling waves","spiral waves brain phase singularity"],"num_per_query":1}

## Code and resource catalog

This section corresponds to `docs/reference/codelib/libraries/travelling_waves_tools.md`.

### Production-worthy or strongly inspirational references

**NeuroPattToolbox (MATLAB).** Implements detection/analysis/visualization of spatiotemporal patterns including wave classification using velocity fields and critical point analysis, with sample data and surrogate/noise-driven verification pathways. It is a direct reference implementation for “pattern classes beyond plane waves.” citeturn24view2turn13search5

**OFAMM / Optical Flow Analysis Toolbox (MATLAB + C++).** Provides optical-flow-based velocity field estimation and wave characterization, explicitly targeting cortical travelling waves (speed/direction/trajectory) in widefield imaging-like data. It is linked to the NeuroImage paper that compares Horn–Schunck, combined local–global, temporospatial methods. citeturn24view3turn12view0turn13search1

**scikit-image optical flow implementations (Python).** `skimage.registration` includes pyramidal iterative Lucas–Kanade flow and TV-L1 flow, giving `cogpy` a lightweight, widely used dependency option for optical flow without custom solvers. citeturn13search3turn13search6

### Reusable “paper code” for travelling-wave analyses (often archival but valuable)

**Generalized Phase (MATLAB).** Provides a phase-estimation method intended to stabilize analytic-signal phase estimates in broadband signals by centering the complex representation and correcting negative-frequency components; tightly linked to Davis et al. (Nature 2020). It also documents dependencies commonly used in wave work (CircStat, etc.). citeturn24view0turn5search21turn5search5

**Working-memory travelling waves code (MATLAB).** The repository accompanying Bhattacharya et al. (PLOS Comp Biol 2022) contains scripts for classifying planar vs rotating waves and computing wave-direction trends using circular–circular correlations and related coefficients. citeturn24view1turn20view0

**Travelling waves vs sequential modules code (mixed; simulation + wave analysis).** Provides analysis and simulation components explicitly designed to test whether observed travelling-wave signatures can arise from sequentially activated discrete modules—useful as a validation adversary / null model reference. citeturn3search8turn6search21turn2search28

### Enabling spectral/time-frequency toolkits (not wave-specific, but essential)

**MNE-Python multitaper PSD (Python).** Provides a stable multitaper PSD function (`psd_array_multitaper`) and is a reasonable dependency for time-domain multitaper building blocks (even if it does not provide travelling-wave logic directly). citeturn7search22

**Nitime multitaper examples and spectral modules (Python).** Provides multitaper spectral estimation examples and code, useful as a reference for multitaper implementation details and testing. citeturn7search2turn7search10

**Prerau Lab multitaper_toolbox (MATLAB/Python/R).** Implements multitaper spectrogram analysis and is relevant as a reference for time-frequency multitaper implementation patterns and parameterization. citeturn4search29

**FieldTrip spectral analysis with multitapers (MATLAB).** FieldTrip documents multitaper-based spectral analysis in its workshop/tutorial material; while not wave-specific, it is a canonical reference for neuroscience multitaper usage. citeturn7search3turn8search21

## Technical design spec

This section corresponds to `docs/specs/travelling_waves_module.md`.

### Design goals derived from the method survey

1. **Multiple complementary estimators** rather than a single “best” travelling-wave detector, because planar/rotational/multi-wave and oscillatory/event-like cases require different assumptions. citeturn13search5turn20view0turn7search7turn14view0  
2. **First-class uncertainty/robustness metrics** (PGD-like fit quality, spectral peak sharpness, bootstrap intervals) because wave claims are sensitive to noise and preprocessing. citeturn3search2turn24view2turn4search35turn9search13  
3. **Geometry-aware API**: support both (a) regular x–y grids and (b) irregular coordinates (SEEG, sparse arrays). This is required for spatial spectra and for fitting wave models on scattered layouts. citeturn23search5turn23academia40turn10view1  
4. **Nonstationarity support** via windowed analysis and event detection, aligning with both oscillatory and non-oscillatory travelling-wave protocols. citeturn7search7turn9search22turn13search5  
5. **Validation-first architecture**: synthetic data generation + surrogate testing included in-core, reflecting the practice of toolboxes that validate against noise-driven surrogates. citeturn24view2turn13search5turn6search21

### Proposed module layout and core abstractions

The architecture below matches the evidence that wave analysis naturally decomposes into spectral, phase, motion, fitting, and validation, while keeping shared representations consistent.

**Top-level package.**
- `cogpy.travelling_waves`
  - `core/`
    - `data_models.py`: typed containers for geometry, estimates, and metadata.
    - `preprocess.py`: filtering, analytic-signal transforms (Hilbert) and broadband-phase options.
    - `windows.py`: windowing, event segmentation.
  - `phase/`
    - `analytic_phase.py`: Hilbert phase; wrappers for generalized phase.
    - `phase_gradient.py`: gradient estimation on grids and scattered points; PGD and circular stats.
    - `phase_singularities.py`: winding-number/topological-charge detectors for spirals.
  - `fitting/`
    - `plane_wave_fit.py`: circular–linear regression plane fits; speed/direction extraction (Zhang-style).
    - `delay_surface.py`: lag estimation + plane/surface fit; robust regression options for small grids.
  - `spectral/`
    - `kw_spectrum.py`: k–ω estimation for regular grids (3D FFT) + ridge/peak extraction.
    - `fk_beamforming.py`: beamforming and Capon-style f–k scanning for arbitrary coordinates.
    - `multitaper_nd.py`: separable multidimensional DPSS tapers (Hanssen-style) integrated with k–ω and f–k.
  - `motion/`
    - `optical_flow.py`: wrappers around scikit-image flow solvers; flow on amplitude/phase/complex fields.
    - `vector_field_features.py`: divergence/curl/critical points; pattern classification hooks.
  - `decomp/`
    - `cpca.py`: complex PCA on analytic-signal maps (optional).
    - `dmd.py`: dynamic mode decomposition (optional).
  - `statespace/`
    - `wave_state_tracker.py`: filtering/smoothing over time for direction/speed; optional switching regimes.
  - `simulation/`
    - `synthetic.py`: generate plane waves, spirals, wave packets, multi-wave mixtures, with noise models.
  - `validation/`
    - `surrogates.py`: phase randomization, time shuffles, spatial shuffles, noise-driven synthetic controls.
    - `metrics.py`: fit quality (PGD-like), spectral peakness, coherence, reproducibility across trials.
  - `viz/`
    - `plots.py`: phase maps, quiver fields, k–ω slices, direction histograms; minimal but method-linked.

This layout is consistent with what current toolboxes emphasize (velocity fields + critical points for pattern classes; and phase-gradient plane fits for direction/speed). citeturn24view2turn13search5turn3search2turn9search22

### API design principles

**Inputs.**
- Accept `numpy.ndarray` and `xarray.DataArray` with explicit dimension labels for `time` and either `space` or `x,y` grid.
- Accept a `Geometry` object: either `(x,y)` grid spacing or per-channel coordinates. This is necessary for both phase-gradient fitting and beamforming-style f–k. citeturn4search8turn23search5turn23academia40

**Outputs.**
- Standardize on a `WaveEstimate` record with:
  - `direction` (angle in radians or unit vector),
  - `speed` (m/s or units of coordinate/time),
  - `frequency` (Hz) and optionally `wavenumber` (rad/m),
  - `wavelength`,
  - `pattern_type` (planar / rotating / spiral / source / sink / mixed / uncertain),
  - `confidence` / uncertainty intervals,
  - `fit_quality` (PGD-like, residual dispersion),
  - `support_mask` (where on grid estimate applies). citeturn3search2turn13search5turn14view0turn17view0

**Method-level contracts.**
- Every detector must implement:
  - `fit(data, geometry, *, window, freq_band, ...) -> WaveEstimate`
  - `score(data, estimate) -> metrics`
  - `plot_diagnostics(...)` (optional, lightweight).
- Every method must ship with synthetic tests that recover known direction/speed and report bias/variance against noise and grid size; this mirrors the simulation-supported style in OFAMM and in wave-simulation adversarial codebases. citeturn12view0turn24view3turn6search21

### Dependency strategy

**Phase 1 dependencies (lightweight).**
- NumPy/SciPy for filtering, DPSS, FFT; xarray optional but supported.
- scikit-image optional extra for optical flow wrappers. citeturn13search3turn13search6

**Optional extras.**
- MNE or nitime as references for multitaper semantics, but `cogpy` should avoid hard dependency unless already used elsewhere; using their APIs as validation references is still useful. citeturn7search22turn7search2

## Prioritized implementation roadmap

This section corresponds to `docs/roadmaps/travelling_waves_methods_implementation.md`.

### Ranking criteria for `cogpy.travelling_waves`

The ranking below is optimized for PixECoG-like grids and spatially arranged LFP/MEA data: interpretability, robustness, implementation tractability, synthetic validation readiness, and literature support.

### Recommended Phase 1 methods

**Phase-gradient plane-wave fitting with PGD-style robustness (oscillatory travelling waves).**  
Why: This is the single most canonical and implementable travelling-wave estimator in invasive electrophysiology for direction and speed from band-limited oscillations, with explicit goodness-of-fit metrics (PGD) and clear synthetic validation routes. It is directly supported by cornerstone ECoG and motor cortex papers and is used in later work as a methodological template. citeturn3search2turn9search22turn2search0turn2search16  
Implementation scope:
- Hilbert analytic phase; spatial unwrapping; circular–linear plane fit; direction/speed; PGD/fit-quality; bootstrap CI.

**k–ω / f–k spectral estimation suite (regular-grid FFT + optional beamforming for irregular geometries).**  
Why: Spectral/array methods provide an orthogonal line of evidence to phase-plane fits and are mature in adjacent fields; they are also the natural home for uncertainty tools via multitapering and for diagnosing multi-wave mixtures (multiple peaks). Minimum viable implementation can start with FFT-based k–ω for regular grids, then add beamforming scans for irregular arrays. citeturn9search1turn4search8turn10view1turn9search13  
Implementation scope:
- 3D FFT windows; peak/ridge extraction → direction/speed/wavelength; coherence/peakness metrics; optional Capon beamforming; separable multidimensional DPSS tapers (Hanssen) as an “enhanced” mode.

**Optical-flow velocity fields on phase/amplitude maps (pattern kinematics + complex morphologies).**  
Why: Optical flow is one of the few families that naturally extends from planar propagation to rotating/spiral/source–sink patterns, and it has explicit neuroscience toolboxes for mesoscale imaging and pattern-class frameworks. A Phase 1 implementation can be thin: wrap established solvers (TV-L1 / iterative Lucas–Kanade) and implement derived features (divergence/curl, critical points). citeturn12view0turn24view2turn13search3turn13search6  
Implementation scope:
- Flow on unwrapped phase or complex analytic components; velocity-field features; minimal pattern classification heuristics; diagnostic plots.

### Recommended Phase 2 methods

**Phase singularity and spiral-center detection (topological charge / winding number).**  
Why: The importance of spirals is growing across modalities, and phase singularities are definitional objects for spirals. The algorithms are mature in excitable media and increasingly used in neuroscience spiral analyses; implementing them enables robust spiral detection and tracking beyond heuristic curl thresholds. citeturn14view0turn17view0turn5search4turn5search12  

**Delay-surface / cross-correlation lag regression for non-oscillatory travelling events.**  
Why: Complements phase-based oscillatory methods and aligns with explicit regression protocols for travelling waves in microelectrode arrays; provides a wave detector for sharp transients and burst-like propagations where phase is unstable. citeturn7search7turn8search3turn4search8  

**Complex PCA and DMD as optional decompositional lenses.**  
Why: They provide compact summaries and can separate modes that resemble travelling components, but they require careful interpretation and surrogate testing. CPCA already has open code in neuroimaging contexts; DMD has extensive theory and is broadly transferable. citeturn5search2turn5search30turn5search39turn5search3  

**State-space smoothing / switching wave states (Kalman / SLDS wrappers).**  
Why: Likely valuable for state-dependent direction switching and transient “wave episodes,” but best implemented after stable per-window estimators exist, using those as observations for a probabilistic tracker. citeturn6search4turn20view0turn11search30  

### Not recommended for now

**End-to-end deep learning optical flow or deep wave classifiers as primary methods.**  
Reason: High validation burden, dependence on large labeled datasets or synthetic realism, and lower interpretability relative to classical estimators—misaligned with a foundational method package. (Deep learning can be revisited after the synthetic validation suite is mature.) This is consistent with toolboxes and papers that emphasize explicit optical-flow/phase-gradient methods rather than black-box detectors. citeturn12view0turn13search5turn13search3  

**Highly specialized MEG/EEG forward-model-based travelling-wave inference as a core package feature.**  
Reason: Source modelling + sensor mixing issues complicate “travelling wave” claims, and parts of the literature emphasize ambiguity between true cortical waves and mixtures of sources at the sensor level; this can be supported later as an application layer, but it is not a good early core for `cogpy.travelling_waves`. citeturn8search12turn7search11turn2search28  

### Concrete Phase 1 deliverables checklist

A Phase 1 that is both scientifically defensible and software-complete should ship:
- A unified `WaveEstimate` object and geometry handling for grids + scattered layouts. citeturn4search8turn23search5  
- Three core estimators (plane-fit PGD; k–ω/f–k; optical flow) + per-estimator confidence metrics. citeturn3search2turn10view1turn13search3turn24view2  
- A synthetic generator for plane waves and rotating/spiral patterns (even if spiral detection is Phase 2) to allow adversarial testing of false positives. citeturn6search21turn14view0turn12view0  
- Surrogate testing utilities (phase randomization / noise-driven controls), reflecting established toolbox practice. citeturn24view2turn13search5