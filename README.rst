TODO: multislice
correct probe or object first?
See https://realpython.com/documenting-python-code/
multigrid/multi-resolution reconstruction strategy
HIO and DM related to ADMM
apply mask in fourier domain
constrain hot pixels
constrain probe counts
minibatch strategies: use sparse set for faster convergence, use compact batch for higher overall quality
use "batch size" ~10-15% of the total number of diffraction patterns
baseline is ptychoshelves: two modes (1) keep data on GPU, (2) reload each iteration
getting started documentation and tutorial. focus on non-expert users (i.e., knows ptychography but not ptychopack)
natural gradient descent
ptycho fourier error
algorithms: fourier vs object domain constraints, TV approach
alternating projection algorithms vs gradient descent algorithms
Fourier domain phase ramp removal?
time-limited reconstructions
probe shifts lead to object shifts
online/live reconstruction support
probe is wavefield prior to interaction with sample; exit wave after interaction
gaussian mixture model of Zernike modes to model probe; decompose into spatial/temporal variations

"during reconstruction the probe can be periodically propagated to the detector
plane (without the influence of the object) where it can be constrained by the
correct free-space intensity"

"it is always advisable to scale the first estimate of the probe by the
integrated intensity in the detector plane. if there is a large disparity
between the intensity of the physical probe and the first guess of the
estimated probe, many reconstruction algorithms find it very hard to recover.
if the edge of the field of view of the reconstruction is very bright or very
dark, you have probably made a mistake."

multimodal probes = shared + spatially varying
- number of shared modes ~ beam coherence
- shared modes: incoherent sum; partial coherence
- spatially varying modes: coherent sum, OPR
- eigenprobes: need eigenweights (for each position) for spatially varying probes
- probe[x,y,shared modes,variable modes]
- mixed state method deals with partial coherence

let me try to be clear the two methods of using different probe modes. the
first one is called mixed-state method to deal with partial coherence, while
the second is PC method to deal with the beam variation. for the case of the
fully coherent beam, then one beam function (single-state) can be used to
represent the beam; however, this fully coherent beam may slightly change, but
in each individual measurement, it can still be considered fully coherent. Then
the PCA method is introduced to deal with this slightly changed probe. Then
going to the second scenario, the beam is partially coherent, then a few
orthogonal probe modes are used in the mixed-state method to deal with the
partial coherence. If this partially coherent beam also varies during
measuremnt, PCA is then introduced to deal with this issue. In principle, PCA
should be applied on all probe modes which are obtained from the mixed-state
method, but PSI code only applies it on the first primary probe mode as it has
the highest power percentage among these mixed states (e.g., > 80%), and the
variation contribution from other mixed-states is little. - Junjing

position correction
- algorithms: annealing, cross-correlation, intensity gradient, image-shift method
- constrain center of bounding box
- no probe information in regions where there is no object to modulate beam (flat regions)
