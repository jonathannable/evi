library(torch)
library(R6)

# High potential -> low probability
# Low potential ->  high probability

# Where does potential come from and why do I mention it vs target or densitiy
# in Wasserstein gradient flow theory, 
# a potential (or functional) F(p) determines 
# the evolution of the probability density p(x,t).

# 
# Abstract Base Potential - negative log target
# 

# abstract base potential
# all target distributions must implement 'compute(x)' which returns
# the potential energy v(x) (where p(x) ~ exp(-v(x))).
PotentialBase <- R6Class("PotentialBase",
  public = list(
    compute = function(x) {
      stop("Method 'compute' must be implemented.")
    }
  )
)

# 
# Donut Potential ----
# 

DonutPotential <- R6Class("DonutPotential",
  inherit = PotentialBase,
  public = list(
    radius = NULL,
    scale = NULL,
    
    initialize = function(radius = 2.0, scale = 0.5) {
      self$radius <- radius
      self$scale <- scale
    },
    
    compute = function(x) {
      # V(x) = scale * (||x|| - radius)^2
      dist <- torch_norm(x, dim = 2)
      return(self$scale * (dist - self$radius)^2)
    }
  )
)

# 
# Banana Potential ----
# 

BananaPotential <- R6Class("BananaPotential",
  inherit = PotentialBase,
  public = list(
    # Using V(x) = 0.5 * x1^2 + 0.5 * (10 * x2 + 3 * x1^2 - 3)^2
    
    compute = function(x) {
      x1 <- x[, 1]
      x2 <- x[, 2]
      
      term1 <- 0.5 * x1^2
      term2 <- 0.5 * (10 * x2 + 3 * x1^2 - 3)^2
      
      return(term1 + term2)
    }
  )
)

# 
# Sine Wave Potential ---- 
# 

SinePotential <- R6Class("SinePotential",
  inherit = PotentialBase,
  public = list(
    scale_y = NULL,
    
    initialize = function(scale_y = 0.4) {
      self$scale_y <- scale_y
    },
    
    compute = function(x) {
      x1 <- x[, 1]
      x2 <- x[, 2]
      mean_curve <- torch_sin(pi * x1 / 2)
      z <- (x2 - mean_curve) / self$scale_y
      # V(x)
      return(0.5 * z^2)
    }
  )
)

# Gaussian Mixture Model GMM ----

GMMPotential <- R6Class("GMMPotential",
  inherit = PotentialBase,
  public = list(
    means = NULL,
    stdev = NULL,
    
    initialize = function(means_matrix, stdev = 0.8) {
      # convert standard r matrix to tensor immediately
      self$means <- torch_tensor(means_matrix, dtype = torch_float())
      self$stdev <- stdev
    },
    
    compute = function(x) {
      # x: (N, 2)
      # means: (K, 2)
      
      # calc distance matrix (n, k)
      dists <- torch_cdist(x, self$means)
      
      # gaussian exponent components: -||x - mu||^2 / (2*sigma^2)
      sq_dists <- dists^2
      log_probs_component <- -sq_dists / (2 * self$stdev^2)
      
      # LogSumExp trick
      # we want v(x) = -log( sum( exp(components) ) )
      # torch_logsumexp computes log(sum(exp(...)))
      log_prob_total <- torch_logsumexp(log_probs_component, dim = 2)
      
      return(-log_prob_total)
    }
  )
)




# 
# Examples
# 

# --- Setup GMM ---
# Triangle Layout
# means_triangle <- matrix(c(
#   0,  2.0,   # Top
#   -2, -1.5,  # Left
#   2, -1.5    # Right
# ), ncol = 2, byrow = TRUE)
# 
# gmm_potential <- GMMPotential$new(means_triangle, stdev = 0.6)
# 
# # --- B. Setup Donut ---
# donut_potential <- DonutPotential$new(radius = 2.5)
# 
# # --- C. Setup Banana ---
# banana_potential <- BananaPotential$new()
# 
