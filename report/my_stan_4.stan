data {
  int<lower=2> T;            // num observations
  int<lower=0> N_test;
  vector[T] y;               // observed outputs
}

parameters {
  vector[2] phi;             // autoregression coeff
  vector[2] theta;                // moving avg coeff
  real<lower=0> sigma;       // noise scale
  //vector[T] err;
  //vector[T] nu;
}

transformed parameters {
  vector[T] err;
  vector[T] nu;
  nu[1] = 0;     // assume err[0] == 0
  err[1] = y[1] - nu[1];
  nu[2] = phi[1] * nu[1] + theta[1] * err[1];
  err[2] = y[2] - nu[2];
  
  for (t in 3:T) {
    nu[t] = phi[1] * nu[t - 1] + phi[2] * nu[t - 2] + theta[1] * err[t - 1] + theta[2] * err[t - 2];
    err[t] = y[t] - nu[t];
  }
}

model {
  //vector[T] nu;              // prediction for time t
  //vector[T] err;             // error for time t
  
  
  phi ~ normal(0, 2);        // priors
  theta ~ normal(0, 2);
  sigma ~ cauchy(0, 5);
  err ~ normal(0, sigma);    // error model
}

generated quantities {
  vector[N_test] predictions;
  
  predictions[1] = phi[1] * nu[T] + phi[2] * nu[T-1] + theta[1] * err[T] + theta[2] * err[T-1];
  
  predictions[2] = phi[1] * predictions[1] + phi[2] * nu[T] + theta[2] * err[T];

  for (k in 3:N_test) {
    predictions[k] = phi[1] * predictions[k-1] + phi[2] * predictions[k-2];
  }
}
