"""Static report prose for the ANOVA statistics pipeline."""

METHODS_TEXT = """\
METHODS (psychophysics statistics)
----------------------------------
Task and response coding. Participants performed a two-alternative
forced-choice (2AFC) stiffness-discrimination task. On each trial a
standard stimulus (S = 85) was compared with one of eight comparison
stimuli (25, 40, 55, 70, 100, 115, 130, 145); the predictor was the
signed difference delta = C - 85. The binary response was coded y = 1 if
and only if the participant judged the COMPARISON to be stiffer than the
standard (this is not correct/incorrect, not side, not order, and not the
physical ground truth). This coding was performed upstream.

Psychometric model. For each subject and finger a lapse-aware yes/no-style
psychometric function with four fitted parameters (mu, scale, lapse_low,
lapse_high) was fitted upstream (these fits are frozen and were NOT
recomputed here). This is not a textbook 2AFC percent-correct model with a
fixed guess rate; it models the probability that the participant judged the
comparison stiffer than the standard, P(y=1) = lapse_low +
(1 - lapse_low - lapse_high) * F(delta; mu, scale), where delta = C - 85 and
F is a monotonic sigmoid. Fits used psignifit when installed and otherwise a
custom lapse-aware fallback fitter; the fitter used for each fit is recorded
in the data.

Derived measures. From each fit we used two summary measures (the fit is in
comparison-stiffness units, so these are equivalent to delta-space): the
Bias, defined as the point of subjective equality minus the standard
(Bias = PSE - 85, the delta at P(y=1) = 0.5; a value of 0 indicates no
perceptual shift), and the JND, defined as (x75 - x25)/2 where x25 and x75
are the comparison values at which the curve reaches 0.25 and 0.75 (a
measure of sensitivity, where smaller values indicate finer
discrimination). Because of the lapse parameters, x25 and x75 are estimable
only if 0.25 and 0.75 lie within the attainable range
[lapse_low, 1 - lapse_high]; when they do not, the upstream fitter returns a
NaN JND with a fit warning, so a non-estimable JND is flagged rather than
silently treated as valid.

Design. System (L vs N) was a between-subjects factor: each participant
used exactly one system. Finger (I, M, P, R) was a within-subjects
(repeated-measures) factor, with all four fingers measured in every
subject. Subject served as the repeated-measures unit.

Main analysis. Bias and JND were each analysed with a mixed-design
analysis of variance with System as the between-subjects factor and Finger
as the within-subjects factor, testing the main effect of System, the main
effect of Finger, and the System x Finger interaction. Sphericity of the
Finger factor was assessed with Mauchly's test and, where appropriate, the
Greenhouse-Geisser correction was applied. Partial eta-squared (np2) and
generalized eta-squared (ges) are reported as effect sizes.

Why a mixed-design ANOVA and not an ordinary two-way ANOVA. An ordinary
independent two-way ANOVA assumes that all observations are independent.
Because the four finger measurements were obtained from the same subject,
they are correlated (repeated measures); treating them as independent would
violate the independence assumption and mis-estimate the error term. We
therefore used the mixed-design model, which correctly partitions
between-subject and within-subject variance.

Exploratory analysis. As a descriptive supplement we also ran a one-way
ANOVA across the eight combined System x Finger groups. This exploratory
analysis treats the eight cells as independent groups and does NOT model
the repeated-measures structure; it is reported for completeness only and
is not the basis for inference.

Planned contrasts. The L-vs-N system difference was tested within each
finger using independent (between-subjects) t-tests, with Holm correction
across the four finger comparisons.

Robustness. Confidence intervals for cell means and for the L - N
difference per finger were additionally estimated with a subject-level
bootstrap (subjects resampled with replacement within each system, keeping
each subject's four finger rows together; fixed random seed).
"""
