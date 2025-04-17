data {
  int<lower=0> N_train;
  int<lower=0> N_test;
  vector[N_train] y_train;
}

parameters {
  real<lower=0> sigma1;
  real<lower=0> sigma2;
  real slope;
  real intercept;
  vector[N_train] x_train;
}

model {
  sigma1 ~ exponential(0.1);
  sigma2 ~ exponential(0.1);
  slope ~ normal(0,100);
  intercept ~ normal(0,100);
  
  for (i in 2:N_train) {
    x_train[i] ~ normal(intercept + slope * x_train[i-1], sigma1);
  }
  
  for (j in 1:N_train) {
    y_train[j] ~ normal(x_train[j], sigma2);
  }
}

generated quantities {
  vector[N_test] predictions;
  
  predictions[1] = normal_rng(intercept + slope * x_train[N_train], sigma1);

  for (k in 2:N_test) {
    predictions[k] = normal_rng(intercept + slope * predictions[k-1], sigma1);
  }
}
