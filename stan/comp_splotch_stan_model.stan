functions {
  // sparse_car_lpdf is written by Max Joseph
  // see: http://mc-stan.org/users/documentation/case-studies/mbjoseph-CARStan.html
  real sparse_car_lpdf(vector phi, real tau, real alpha, int[,] W_sparse, vector D_sparse, vector lambda, int n, int W_n) {
    row_vector[n] phit_D;
    row_vector[n] phit_W;
    vector[n] ldet_terms;
  
    phit_D = (phi .* D_sparse)';
    phit_W = rep_row_vector(0,n);
    for (i in 1:W_n) {
      phit_W[W_sparse[i,1]] = phit_W[W_sparse[i,1]]+phi[W_sparse[i,2]];
      phit_W[W_sparse[i,2]] = phit_W[W_sparse[i,2]]+phi[W_sparse[i,1]];
    }
    ldet_terms = log1m(alpha*lambda);
    return 0.5*(n*log(tau)+sum(ldet_terms)-tau*(phit_D*phi-alpha*(phit_W*phi)));
  }
}

data {
  int<lower=1> N_tissues; // number of tissue sections
  int<lower=1> N_spots[N_tissues]; // number of spots per tissue section
  int<lower=1> N_covariates; // number of AARs
  int<lower=1> N_celltypes; // number of cell types
  int<lower=1,upper=3> N_levels; // number of levels
  int<lower=1> N_level_1; // number of level 1 variables
  int<lower=0> N_level_2; // number of level 2 variables
  int<lower=0> N_level_3; // number of level 3 variables

  int<lower=0,upper=1> nb;  // whether to use negative binomial (defaults to Poisson)
  int<lower=0,upper=1> zi;  // whether to use zero inflation
  int<lower=0,upper=1> car;

  row_vector[N_celltypes] beta_prior_mean;
  row_vector[N_celltypes] beta_prior_std;

  // level 1 index of each level 2 variable  (this is used for indexing beta_level_1)
  int<lower=1> level_2_mapping[N_level_2]; 
  // level 2 index of each level 3 variable  (this is used for indexing beta_level_2)
  int<lower=1> level_3_mapping[N_level_3]; 
  // level 3 index of each tissue section (this is used for indexing beta_level_3)
  int<lower=1> tissue_mapping[N_tissues]; 

  int<lower=0> counts[sum(N_spots)]; // counts per spot

  vector<lower=0>[sum(N_spots)] size_factors; // size factor for each spot

  // annotation for each spot (this is used for indexing beta_mouse)
  int<lower=1> D[sum(N_spots)]; 
  // cell type composition for each spot
  simplex[N_celltypes] E[sum(N_spots)];

  int<lower=0> W_n[car ? 1 : 0]; // number of adjacent spot pairs
  int W_sparse[car ? W_n[1] : 0,car ? 2 : 0]; // adjacency pairs
  vector[car ? sum(N_spots) : 0] D_sparse; // number of neighbors for each spot
  vector[car ? sum(N_spots) : 0] eig_values; // eigenvalues of D^{-0.5} W D^{-0.5}
}

transformed data {
  // log-transformed size factors are more convenient (poisson_log)
  vector[sum(N_spots)] log_size_factors;
  // cumulative sum of spots over tissue sections (makes indexing a bit easier)
  int<lower=0> csum_N_spots[N_tissues+1];
  // total number of spots
  int<lower=0> sum_N_spots;

  // transform size factors to log space
  log_size_factors = log(size_factors);

  // calculate cumulative sum of spots over tissue sections
  csum_N_spots[1] = 0;
  for (i in 2:(N_tissues+1)) {
    csum_N_spots[i] = csum_N_spots[i-1]+N_spots[i-1];
  }

  // get total number of spots
  sum_N_spots = sum(N_spots);

}

parameters {
  // CAR
  vector[car ? sum_N_spots : 0] psi;

  // non-centered parametrization of coefficients
  row_vector[N_celltypes] beta_raw[N_level_1+N_level_2+N_level_3, N_covariates];

  // conditional precision
  real<lower=0> tau[car ? 1 : 0];
  // spatial autocorrelation
  real<lower=0,upper=1> a[car ? 1 : 0];

  // standard deviation of epsilon (spot-level variation)
  real<lower=0> sigma;

  // standard deviations of levels 2 and 3 in linear model
  real<lower=0> sigma_level_2[N_level_2 ? 1 : 0];
  real<lower=0> sigma_level_3[N_level_3 ? 1 : 0];
  
  // probability of extra zeros
  real<lower=0,upper=1> theta[zi ? 1 : 0];

  // overdispersion parameter for negative binomial
  real<lower=0.000001> phi[nb ? 1 : 0];

  // non-centered parametrization of spot-level variation
  vector[sum_N_spots] noise_raw;
}

transformed parameters {
  // rate parameter
  vector[sum_N_spots] log_lambda;

  // dispersion parameter
  vector[nb ? sum_N_spots : 0] phi_arr;

  // level 1 coefficients
  row_vector[N_celltypes] beta_level_1[N_level_1, N_covariates];
  // level 2 coefficients
  row_vector[N_celltypes] beta_level_2[N_level_2, N_level_2 ? N_covariates : 0];
  // level 3 coefficients
  row_vector[N_celltypes] beta_level_3[N_level_3, N_level_3 ? N_covariates : 0];

  // derive level 1 coefficients from beta_raw
  for (i in 1:N_level_1) {
    for (j in 1:N_covariates) {
      //beta_level_1[i,j] = 2.0*beta_raw[i,j];
      beta_level_1[i,j] = beta_raw[i,j] .* beta_prior_std + beta_prior_mean;
    }
  }

  // derive level 2 coefficients from beta_level_1, sigma_level_2, and beta_raw
  if (N_level_2) {
    for (i in 1:N_level_2) {
      for (j in 1:N_covariates) {
        beta_level_2[i,j] = beta_level_1[level_2_mapping[i],j]
          +sigma_level_2[1]*beta_raw[N_level_1+i,j];
      }
    }
  }

  // derive level 3 coefficients from beta_level_2, sigma_level_3, and beta_raw
  if (N_level_3) {
    for (i in 1:N_level_3) {
      for (j in 1:N_covariates) {
        beta_level_3[i,j] = beta_level_2[level_3_mapping[i],j]
          +sigma_level_3[1]*beta_raw[N_level_1+N_level_2+i,j];
      }
    }
  }

  // derive log_lambda using beta_level_x, psi, sigma, and noise_raw
  // D is used to get correct AAR element from beta_level_x for each spot
  // tissue_mapping is used to get correct beta_level_x vector for each tissue section
  if (N_level_3) {
    if (car) { 
      for (i in 1:N_tissues) {
        for (j in csum_N_spots[i]+1:csum_N_spots[i+1]) {
          log_lambda[j] = beta_level_3[tissue_mapping[i],D[j]] * E[j] 
          + psi[j] 
          + 0.3 * sigma * noise_raw[j];
        }
      }
    } else {
      for (i in 1:N_tissues) {
        for (j in csum_N_spots[i]+1:csum_N_spots[i+1]) {
          log_lambda[j] = beta_level_3[tissue_mapping[i],D[j]] * E[j] 
          + 0.3 * sigma * noise_raw[j];
        }
      }
    }
  } else if (N_level_2) {
    if (car) { 
      for (i in 1:N_tissues) {
        for (j in csum_N_spots[i]+1:csum_N_spots[i+1]) {
          log_lambda[j] = beta_level_2[tissue_mapping[i],D[j]] * E[j]
          + psi[j]
          + 0.3 * sigma * noise_raw[j];
        }
      }
    } else {
      for (i in 1:N_tissues) {
        for (j in csum_N_spots[i]+1:csum_N_spots[i+1]) {
          log_lambda[j] = beta_level_2[tissue_mapping[i],D[j]] * E[j]
          + 0.3 * sigma * noise_raw[j];
        }
      }
    }
  } else if (N_level_1) {
    if (car) { 
      for (i in 1:N_tissues) {
        for (j in csum_N_spots[i]+1:csum_N_spots[i+1]) {
          log_lambda[j] = beta_level_1[tissue_mapping[i],D[j]] * E[j]
          + psi[j]
          + 0.3 * sigma * noise_raw[j];
        }
      }
    } else {
      for (i in 1:N_tissues) {
        for (j in csum_N_spots[i]+1:csum_N_spots[i+1]) {
          log_lambda[j] = beta_level_1[tissue_mapping[i],D[j]] * E[j]
          + 0.3 * sigma * noise_raw[j];
        }
      }
    }
  }

  // broadcast the gene-wide dispersion parameter phi into an array of size sum_N_spots
  // used for likelihood calculation in case of NB distribution
  if (nb) {
    for (i in 1:sum_N_spots) {
      phi_arr[i] = phi[1];
    }
  }
}

model {
  if (car) {
    // parameters of CAR (a has a Uniform(0,1) prior)
    tau ~ inv_gamma(1,1);
    // CAR
    psi ~ sparse_car(tau[1],a[1],W_sparse,D_sparse,eig_values,sum_N_spots,W_n[1]);
  }

  if (zi) {
    // parameter of probability of extra zeros
    theta ~ beta(1,2);
  }

  // spot-level variation
  // non-centered parameterization
  sigma ~ normal(0,1);
  noise_raw ~ normal(0,1);

  // linear model
  // non-centered parameterization
  if (N_level_2)
    sigma_level_2[1] ~ normal(0,1);
  if (N_level_3)
    sigma_level_3[1] ~ normal(0,1);

  for (i in 1:N_level_1+N_level_2+N_level_3) {
    for (j in 1:N_covariates) {
      beta_raw[i,j] ~ normal(0,1);
    }
  }

  if (nb) {
    phi ~ gamma(2,1); // inverse overdispersion (var = mu + mu^2/phi)
    
    if (zi) {
    // zero-inflated negative binomial (ZINB) likelihood
      for (i in 1:sum_N_spots) {
        if (counts[i] == 0)
          target += log_sum_exp(bernoulli_lpmf(1|theta[1]),
                                bernoulli_lpmf(0|theta[1])
                                  + neg_binomial_2_log_lpmf(counts[i]|log_lambda[i]+log_size_factors[i], phi_arr));
        else
          target += bernoulli_lpmf(0|theta[1])
                      + neg_binomial_2_log_lpmf(counts[i]|log_lambda[i]+log_size_factors[i], phi_arr);
      }
    } else {
      counts ~ neg_binomial_2_log(log_lambda+log_size_factors, phi_arr);
    }
  } 
  else {
    if (zi) {
    // zero-inflated Poisson likelihood
      for (i in 1:sum_N_spots) {
        if (counts[i] == 0)
          target += log_sum_exp(bernoulli_lpmf(1|theta[1]),
                                bernoulli_lpmf(0|theta[1])
                                  +poisson_log_lpmf(counts[i]|log_lambda[i]+log_size_factors[i]));
        else
          target += bernoulli_lpmf(0|theta[1])
                      + poisson_log_lpmf(counts[i]|log_lambda[i]+log_size_factors[i]);
      }
    } else {
      counts ~ poisson_log(log_lambda+log_size_factors);
    }
  }
}

generated quantities {
}
