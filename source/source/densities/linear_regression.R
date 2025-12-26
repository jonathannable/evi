library(alr4)
library(tidyverse)

source("./evi_solver.R")
source("./kernels.R")
source("./plotting.R")


# Linear Regression ----

# y = XB+e
# y nx1
# X nxp
# B px1
# e nx1 ~ N(0,sigma^2 I)

# Model: normal-normal

# Likelihood:
# y | beta, sigma^2, X ~ N(Xbeta, sigma^2I)

# Prior
# Beta ~ N(mu_0,gamma_0)
# for a constant mu_0=0, and a known gamma_0.

# Posterior 
# p(Beta | y, X) \proto p(Beta) * p(y|X,Beta)

# Beta | y, X ~ N(mu_n, gamma_n)

# gamma_n^(-1) = gamma_0^(0) + 1/sigma^2 X^T X
# mu_n = gamma_n*(gamma^(-1)mu_0+1/sigma^2X^T y)

# The class defined below assumes a known covariance on the beta prior

# If we place a Inverse Gamma prior on Beta's covariance we get the joint posterior

# Beta | sigma^2_beta ~ IVG(shape=alpha_0, scale=beta_0)

# P(Beta, gamma_0 | y, X) ~ Normal-Inverse-Gamma 

# Marginal conditional distributions
# Beta | y, X ~ T(df=2a_n)
# sigma^2_beta | y, X  ~ InverseGamma(a_n, b_n)

LinearModelPotential <- R6Class("LinearModelPotential",
  inherit = PotentialBase,
  public = list(
    X = NULL,
    y = NULL,
    noise_sigma = NULL,
    prior_sigma = NULL,
    
    initialize = function(X, y, noise_sigma = 1.0, prior_sigma = 1.0) {
      # Data Validation: Ensure inputs are Torch tensors
      self$X <- if (is_torch_dtype(X)) X else torch_tensor(X)
      # self$y <- if (is_torch_dtype(y)) y else torch_tensor(y)
      # y needs to be a vector, but how I set u
      y_clean <- if (is_torch_dtype(y)) as_array(y) else y
      self$y <- torch_tensor(as.numeric(y_clean))      
      
      # Constants
      self$noise_sigma <- noise_sigma
      self$prior_sigma <- prior_sigma
    },
    
    compute = function(x) {
      # In this context, x represents theta (coefficients)
      # Shape of x: (N_particles, D_features)
      
      # 1. Log Prior Term: Gaussian(0, prior_sigma)
      # sum(theta^2) over dimensions (dim=2)
      prior_term <- 0.5 * torch_sum(x^2, dim = 2)$squeeze() / (self$prior_sigma^2)
      
      # 2. Log Likelihood Term
      # Fitted Values: X (M, D) @ theta_t (D, N) -> (M, N)
      
      # Result: y_fit is a matrix of shape (M_samples, N_particles).
      # Column 1 contains the fitted values for the entire dataset using the weights from Particle 1.
      # Column 2 contains the fitted values using weights from Particle 2, etc.      
      
      y_fit <- torch_matmul(self$X, x$t())
    
      # Residuals: (M, N) - (M, 1) broadcasted
      residuals <- y_fit - self$y$unsqueeze(2)
      
      # Sum of Squared Errors (SSE) over data points (dim=1) -> (N)
      sse <- torch_sum(residuals^2, dim = 1)$squeeze()
      
      likelihood_term <- 0.5 * sse / (self$noise_sigma^2)
      
      # Return shape: (N)
      return(prior_term + likelihood_term)
    }
  )
)

# Example ----
# Generate Data ---- 
# simple linear regresion: intercept + 1 slope
n_obs <- 50
p = 2 # parameters 
n_particles <- 100
# X_raw <- matrix(rnorm(n_obs * p), ncol=p)
X_raw <- matrix(rnorm(n_obs * 1, 10, 2), ncol=1)
X_raw <- cbind(rep(1,n_obs),X_raw)
weights_true <- c(-1.5, 2.0)
# X_raw %*% weights_true + rnorm(n_obs, sd=0.5) -> 2d tensor 
# but the compute requires a vector
# as.numeric fixed this...
y_raw <- as.numeric(X_raw %*% weights_true + rnorm(n_obs, sd=0.5))

# Initial state/data
x_init <- torch_randn(n_particles, p)

# EVI Solver Setup ----
target_func <- LinearModelPotential$new(
  X_raw, y_raw, 
  noise_sigma=1, # error term variance 
  prior_sigma=10.0 # coefficient prior variance
)

gauss_kernel <- GaussianKernel$new(h = 1)

stop_rules <- list(StopDisplacement$new(1e-5), StopEnergyPlateau$new())

solver <- EVISolver$new(
  target_neg_log_prob = target_func,
  kernel = gauss_kernel,
  tau = 1e-2    # Small tau - slow accurate movements, high friction: Large tau - big jumps faster convergence
)

solver$fit(x_init, 
           time_steps = 500, 
           inner_steps = 100,
           optimizer = optim_adam, 
           optimizer_config=list(lr=1e-3),
           outer_stop_rules = stop_rules)

# Analysis ---- 
particles <- solver$get_final_particles()
cat(sprintf("True Values: %f, %f", weights_true[1],weights_true[2]))
particle_mean <- colMeans(particles)
particle_median <- apply(particles,2,FUN=median)
cat("Particle Mean",paste(particle_mean))
cat("Particle Median",paste(particle_median))
b0 <- particle_median[[1]]
b1 <- particle_median[[2]]
plot(0, 0, type = "n", xlim = c(0, 10), ylim = c(0, 10),
     xlab = "x", ylab = "y")
abline(a = b0, b = b1, col = "blue", lwd = 2)
abline(a=weights_true[1],b=weights_true[2],col="red",lwd=2)

particles_df = data.frame(b0=particles[,1], b1=particles[,2])

# Posterior Dists for Parameters
ggplot(particles_df)+
  geom_density(aes(x=b0),fill="darkred")+
  geom_point(aes(x=b0,y=0),shape=21,alpha=0.25)+
  geom_vline(xintercept = weights_true[1])

ggplot(particles_df)+
  geom_density(aes(x=b1),fill="lightblue")+
  geom_point(aes(x=b1,y=0),shape=21,alpha=0.25)+
  geom_vline(xintercept = weights_true[2])

ggplot(particles_df,aes(x=b0,y=b1))+geom_density_2d()+geom_point()


analysis <- ParticleDistribution$new(particles_df)
analysis$summary()
