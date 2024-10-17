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
  int<lower=1,upper=3> N_levels; // number of levels
  int<lower=1> N_level_1; // number of level 1 variables
  int<lower=0> N_level_2; // number of level 2 variables
  int<lower=0> N_level_3; // number of level 3 variables

  int<lower=0,upper=1> zi;
  int<lower=0,upper=1> car;

  // level 1 index of each level 2 variable  (this is used for indexing beta_level_1)
  array[N_level_2] int<lower=1> level_2_mapping;  
  // level 2 index of each level 3 variable  (this is used for indexing beta_level_2)
  array[N_level_3] int<lower=1> level_3_mapping; 
  // level 3 index of each tissue section (this is used for indexing beta_level_3)
  array[N_tissues] int<lower=1> tissue_mapping; 

  array[sum(N_spots)] int<lower=0> counts; // counts per spot

  vector<lower=0>[sum(N_spots)] size_factors; // size factor for each spot

  // annotation for each spot (this is used for indexing beta_mouse)
  array[sum(N_spots)] int<lower=1> D; 

  array[car ? 1 : 0] int<lower=0> W_n; // number of adjacent spot pairs
  // array[car ? W_n[1] : 0, car ? 2 : 0] int W_sparse; // adjacency pairs as adj. matrix

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
  matrix[N_level_1+N_level_2+N_level_3,N_covariates] beta_raw;

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

  // non-centered parametrization of spot-level variation
  vector[sum_N_spots] noise_raw;
}

transformed parameters {
  // rate parameter
  vector[sum_N_spots] log_lambda;

  // level 1 coefficients
  matrix[N_level_1,N_covariates] beta_level_1;
  // level 2 coefficients
  matrix[N_level_2,N_level_2 ? N_covariates : 0] beta_level_2;
  // level 3 coefficients
  matrix[N_level_3,N_level_3 ? N_covariates : 0] beta_level_3;

  // derive level 1 coefficients from beta_raw
  for (i in 1:N_level_1) {
    beta_level_1[i] = 2.0*beta_raw[i];
  }

  // derive level 2 coefficients from beta_level_1, sigma_level_2, and beta_raw
  if (N_level_2) {
    for (i in 1:N_level_2) {
      beta_level_2[i] = beta_level_1[level_2_mapping[i]]
        +sigma_level_2[1]*beta_raw[N_level_1+i];
    }
  }

  // derive level 3 coefficients from beta_level_2, sigma_level_3, and beta_raw
  if (N_level_3) {
    for (i in 1:N_level_3) {
      beta_level_3[i] = beta_level_2[level_3_mapping[i]]
        +sigma_level_3[1]*beta_raw[N_level_1+N_level_2+i];
    }
  }

  // derive log_lambda using beta_level_x, psi, sigma, and noise_raw
  // D is used to get correct AAR element from beta_level_x for each spot
  // tissue_mapping is used to get correct beta_level_x vector for each tissue section
  if (N_level_3) {
    if (car) { 
      for (i in 1:N_tissues) {
        log_lambda[csum_N_spots[i]+1:csum_N_spots[i+1]] = 
          beta_level_3[tissue_mapping[i]][D[csum_N_spots[i]+1:csum_N_spots[i+1]]]'
          +psi[csum_N_spots[i]+1:csum_N_spots[i+1]]
          +0.3*sigma*noise_raw[csum_N_spots[i]+1:csum_N_spots[i+1]];
      }
    } else {
      for (i in 1:N_tissues) {
        log_lambda[csum_N_spots[i]+1:csum_N_spots[i+1]] = 
          beta_level_3[tissue_mapping[i]][D[csum_N_spots[i]+1:csum_N_spots[i+1]]]'
          +0.3*sigma*noise_raw[csum_N_spots[i]+1:csum_N_spots[i+1]];
      }
    }
  } else if (N_level_2) {
    if (car) { 
      for (i in 1:N_tissues) {
        log_lambda[csum_N_spots[i]+1:csum_N_spots[i+1]] = 
          beta_level_2[tissue_mapping[i]][D[csum_N_spots[i]+1:csum_N_spots[i+1]]]'
          +psi[csum_N_spots[i]+1:csum_N_spots[i+1]]
          +0.3*sigma*noise_raw[csum_N_spots[i]+1:csum_N_spots[i+1]];
      }
    } else {
      for (i in 1:N_tissues) {
        log_lambda[csum_N_spots[i]+1:csum_N_spots[i+1]] = 
          beta_level_2[tissue_mapping[i]][D[csum_N_spots[i]+1:csum_N_spots[i+1]]]'
          +0.3*sigma*noise_raw[csum_N_spots[i]+1:csum_N_spots[i+1]];
      }
    }
  } else if (N_level_1) {
    if (car) { 
      for (i in 1:N_tissues) {
        log_lambda[csum_N_spots[i]+1:csum_N_spots[i+1]] = 
          beta_level_1[tissue_mapping[i]][D[csum_N_spots[i]+1:csum_N_spots[i+1]]]'
          +psi[csum_N_spots[i]+1:csum_N_spots[i+1]]
          +0.3*sigma*noise_raw[csum_N_spots[i]+1:csum_N_spots[i+1]];
      }
    } else {
      for (i in 1:N_tissues) {
        log_lambda[csum_N_spots[i]+1:csum_N_spots[i+1]] = 
          beta_level_1[tissue_mapping[i]][D[csum_N_spots[i]+1:csum_N_spots[i+1]]]'
          +0.3*sigma*noise_raw[csum_N_spots[i]+1:csum_N_spots[i+1]];
      }
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

  to_vector(beta_raw) ~ normal(0,1);

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
    counts ~ poisson_log(log_lambda+log_size_factors);
  }
}

generated quantities {
}
