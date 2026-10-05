functions {
  // sparse_car_lpdf is written by Max Joseph
  // see: http://mc-stan.org/users/documentation/case-studies/mbjoseph-CARStan.html
  real sparse_car_lpdf(
    vector phi,
    real tau,
    real alpha,
    array[] int V, // CSR indices
    array[] int U, // CSR indptr
    vector W,  // CSR data
    vector D_sparse, // Number of Adjacencies
    vector lambda, // Eigenvalues
    int n
  ) {

    real lpdf = n * log(tau);
    real phi_prec_phi;

    lpdf += sum(log1m(alpha * lambda));
  
    phi_prec_phi = (phi .* D_sparse)' * phi;
    phi_prec_phi -= csr_matrix_times_vector(n, n, W, V, U, phi)' * phi * alpha;

    return 0.5 * (lpdf - tau * phi_prec_phi);
  }

  int numzero(array[] int x, int n_x) {
    int nz = 0;
    for (i in 1:n_x) {
      nz += (x[i] == 0);
    }
    return nz;
  }
}

data {
  int<lower=1> N_tissues; // number of tissue sections
  array[N_tissues] int<lower=1> N_spots; // number of spots per tissue section
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
  array[N_level_2] int<lower=1> level_2_mapping; 
  // level 2 index of each level 3 variable  (this is used for indexing beta_level_2)
  array[N_level_3]int<lower=1> level_3_mapping; 
  // level 3 index of each tissue section (this is used for indexing beta_level_3)
  array[N_tissues]int<lower=1> tissue_mapping; 

  array[sum(N_spots)]int<lower=0> counts; // counts per spot

  vector<lower=0>[sum(N_spots)] size_factors; // size factor for each spot

  // annotation for each spot (this is used for indexing beta_mouse)
  array[sum(N_spots)] int<lower=1> D;

  // cell type composition for each spot
  array[sum(N_spots)] simplex[N_celltypes] E;

  array[car ? 1 : 0] int<lower=0> W_n; // number of adjacent spot pairs
  array[car ? W_n[1] * 2 : 0] int<lower=1> V; // adjacency pairs as a CSR - indices
  array[car ? sum(N_spots) + 1 : 0] int<lower=0> U; // adjacency pairs as a CSR - indptr

  vector[car ? sum(N_spots) : 0] D_sparse; // number of neighbors for each spot
  vector[car ? sum(N_spots) : 0] eig_values; // eigenvalues of D^{-0.5} W D^{-0.5}
}

transformed data {
  // log-transformed size factors are more convenient (poisson_log)
  vector[sum(N_spots)] log_size_factors;
  // cumulative sum of spots over tissue sections (makes indexing a bit easier)
  array[N_tissues+1] int<lower=0> csum_N_spots;
  // total number of spots
  int<lower=0> sum_N_spots;
  // number of zero values
  int<lower = 0> N_zero = numzero(counts, sum(N_spots));
  int<lower = 0> N_nonzero = sum(N_spots) - N_zero;
  array[N_zero] int<lower=0> zero_pointers;
  array[N_nonzero] int<lower=0> nz_pointers;
  array[N_nonzero] int<lower=0> nz_counts;

  vector[car ? W_n[1] * 2 : 0] W = rep_vector(1, car ? W_n[1] * 2 : 0); // adjacency pairs as a CSR - data

  // transform size factors to log space
  log_size_factors = log(size_factors);

  // calculate cumulative sum of spots over tissue sections
  csum_N_spots[1] = 0;
  for (i in 2:(N_tissues+1)) {
    csum_N_spots[i] = csum_N_spots[i-1]+N_spots[i-1];
  }

  // get total number of spots
  sum_N_spots = sum(N_spots);

  // get an array of pointers to the zeros
  // an array of pointers to the nonzeros
  // and the nonzero values themselves in another array
  int zero_pointer = 0;
  int nz_pointer = 0;
  for (i in 1:sum_N_spots) {
    if (counts[i] == 0) {
      zero_pointer += 1;
      zero_pointers[zero_pointer] = i;
    }
    else {
      nz_pointer += 1;
      nz_pointers[nz_pointer] = i;
      nz_counts[nz_pointer] = counts[i];
    }
  }
}

parameters {
  // CAR
  vector[car ? sum_N_spots : 0] psi;

  // non-centered parametrization of coefficients
  array[N_level_1+N_level_2+N_level_3, N_covariates] row_vector[N_celltypes] beta_raw;

  // conditional precision
  array[car ? 1 : 0] real<lower=0> tau;
  // spatial autocorrelation
  array[car ? 1 : 0] real<lower=0,upper=1> a;

  // standard deviation of epsilon (spot-level variation)
  real<lower=0> sigma;

  // standard deviations of levels 2 and 3 in linear model
  array[N_level_2 ? 1 : 0] real<lower=0> sigma_level_2;
  array[N_level_3 ? 1 : 0] real<lower=0> sigma_level_3;
  
  // probability of extra zeros
  array[zi ? 1 : 0] real<lower=0,upper=1> theta;

  // overdispersion parameter for negative binomial
  array[nb ? 1 : 0] real<lower=0.000001> phi;

  // non-centered parametrization of spot-level variation
  vector[sum_N_spots] noise_raw;
}

transformed parameters {
  // rate parameter
  vector[sum_N_spots] log_lambda;

  // dispersion parameter
  vector[nb ? sum_N_spots : 0] phi_arr;

  // level 1 coefficients
  array[N_level_1, N_covariates] row_vector[N_celltypes] beta_level_1;
  // level 2 coefficients
  array[N_level_2, N_level_2 ? N_covariates : 0] row_vector[N_celltypes] beta_level_2;
  // level 3 coefficients
  array[N_level_3, N_level_3 ? N_covariates : 0] row_vector[N_celltypes] beta_level_3;

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
    psi ~ sparse_car(tau[1], a[1], V, U, W, D_sparse, eig_values, sum_N_spots);
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
      // zero-inflated Negative Binomial likelihood
      vector[sum_N_spots] log_lambda_size = log_lambda + log_size_factors;
      real heads = bernoulli_lpmf(1|theta[1]);
      real tails = bernoulli_lpmf(0|theta[1]);

      // loop through zeros
      for (i in 1:N_zero) {
        target += log_sum_exp(heads, tails + neg_binomial_2_log_lpmf(0|log_lambda_size[zero_pointers[i]], phi_arr[zero_pointers[i]]));
      }

      // vectorized from index
      target += N_nonzero * tails;
      target += neg_binomial_2_log_lpmf(nz_counts|log_lambda_size[nz_pointers], phi_arr[nz_pointers]);
    } 
    else {
      // Negative Binomial likelihood
      counts ~ neg_binomial_2_log(log_lambda+log_size_factors, phi_arr);
    }
  } 
  else {
    if (zi) {
      // zero-inflated Poisson likelihood
      vector[sum_N_spots] log_lambda_size = log_lambda + log_size_factors;
      real heads = bernoulli_lpmf(1|theta[1]);
      real tails = bernoulli_lpmf(0|theta[1]);

      // loop through zeros
      for (i in 1:N_zero) {
        target += log_sum_exp(heads, tails + poisson_log_lpmf(0|log_lambda_size[zero_pointers[i]]));
      }

      // vectorized from index
      target += N_nonzero * tails;
      target += poisson_log_lpmf(nz_counts|log_lambda_size[nz_pointers]);
    } 
    else {
      // Poisson likelihood
      counts ~ poisson_log(log_lambda+log_size_factors);
    }
  }
}

generated quantities {
}
