library(torch)
library(R6)

# 
# Kernel Interface & Implementations
# 

# Defines Two Kernels: GaussianKernel, LaplacianKernel.

# Abstract Base Kernel
# Defines the contract that all kernels must follow.
KernelBase <- R6Class("KernelBase",
  public = list(
    compute = function(x, y) {
      stop("Method 'compute' must be implemented by the subclass.")
    }
  )
)

# Setup a couple Kernels: RBF is common

# Gaussian (RBF) Kernel
# K(x,y) = C * exp(-||x-y||^2 / h^2)
GaussianKernel <- R6Class("GaussianKernel",
  inherit = KernelBase,
  public = list(
    h = NULL,
    
    initialize = function(h = 1.0) {
      self$h <- h
    },
    
    compute = function(xi, xj) {
      # Expects inputs of shape: xi (N, 1, d), xj (1, N, d) for broadcasting
      diff <- xi - xj
      squared_norm <- torch_sum(diff^2, dim = 3)
      
      d <- xi$size(3)
      # Normalizer calculation (pre-calculated or done on fly)
      # Note: In high dimensions, this term can become very small.
      normalizer <- 1 / ((sqrt(2 * pi) * self$h)^d)
      
      return(normalizer * torch_exp(-squared_norm / (self$h^2)))
    }
  )
)

# Laplacian Kernel
# Example of a different kernel someone might want to swap in.
# K(x,y) = C * exp(-||x-y|| / h)
LaplacianKernel <- R6Class("LaplacianKernel",
  inherit = KernelBase,
  public = list(
    h = NULL,
    initialize = function(h = 1.0) { self$h <- h },
    
    compute = function(xi, xj) {
      diff <- xi - xj
      # L1 norm or L2 norm depending on definition. 
      # Standard Laplacian kernel often uses L1 distance (Manhattan) 
      # but in RBF context often L2 distance is used with exp(-d/h).
      # Using L2 distance here for rotation invariance:
      dist <- torch_norm(diff, p = 2, dim = 3)
      
      return(torch_exp(-dist / self$h))
    }
  )
)