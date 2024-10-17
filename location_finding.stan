data {
  int<lower=1> T;  // Number of measurement points
  int<lower=1> p;  // Dimension of the data
  array[T] real y; // outcome y is 1D
  array[T] vector[p] x; // data is x is pD
  int<lower=1> K;  // Number of sources
  real<lower=0> noise_sd;  // noise standard deviation
  real<lower=0> base_signal;  // base signal
  real<lower=0> max_signal;  // max signal
}

parameters {
  array[K] vector[p] theta;  // 2D locations of the K sources
}

model {
  // Priors for theta
  for (k in 1:K) {
    theta[k] ~ normal(0, 1);  // standard normal prior
  }

  // Likelihood
  for (t in 1:T) {
    vector[K] sq_two_norm;
    for (k in 1:K) {
      sq_two_norm[k] = dot_self(x[t] - theta[k]);
    }
    real mean_y = log(base_signal + sum(inv(max_signal + sq_two_norm)));
    y[t] ~ normal(mean_y, noise_sd);
  }
}
