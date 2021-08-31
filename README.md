# logistic-fit
Script for fitting a convex combination of logistic modes to a given sample of observations.

estimate tries to optimize the mean square distance between the sample CDF and the logistic CDF, estimate_log does maximum likelihood on the given sample using BFGS, and estimate_powell does maximum likelihood using the Powell optimizer. Runtime is on the order of 10 seconds on free Colab for 4-5 modes and sample sizes of around 1000.

For N modes, the estimate functions return a list of 3N - 1 numbers

\[M1, D1, M2, D2, ..., MN, DN, W2, W3, ..., WN\]

where Mk is the median of the kth logistic mode, Dk is the distance from the median to the 75th percentile point of the kth logistic mode, and Wk is the softmax weight of the kth logistic mode. In other words, the actual weight of the kth mode in the convex combination is given by exp(Wk)/(sum of exp(Wi) over all modes i) where W1 = 0 by assumption.
