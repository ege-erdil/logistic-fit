# logistic-fit
Script for fitting a convex combination of logistic modes to a given sample of observations.

estimate tries to optimize the mean square distance between the sample CDF and the logistic CDF, estimate_log does maximum likelihood on the given sample using BFGS, and estimate_powell does maximum likelihood using the Powell optimizer. Runtime is on the order of 10 seconds on free Colab for 4-5 modes and sample sizes of around 1000.
