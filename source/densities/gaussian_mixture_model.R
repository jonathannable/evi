library(torch)

# Factor function to generate density
make_gmm_density <- function(means_matrix, stdev = 0.8) {
  # means_matrix: a K x 2 matrix (K modes, 2 dimensions)
  # stdev: standard deviation of the blobs
  
  # Convert parameters to tensors once to save time
  means_tensor <- torch_tensor(means_matrix, dtype = torch_float())
  
  function(x) {
    # x shape: [batch_size, 2]
    # means shape: [n_modes, 2]
    
    # calculate squared euclidean distance between every x and every mean
    # Rely on torch broadcasting or manual expansion. 
    # Use torch_cdist for efficiency:
    # input x: (N, 2), means: (K, 2) -> dists: (N, K)
    dists <- torch_cdist(x, means_tensor)
    
    # square the distances for gaussian exponent
    sq_dists <- dists^2
    
    # calculate log-probability for each mode (unnormalized)
    # log(exp(-dist^2 / 2s^2)) = -dist^2 / (2s^2)
    # assume equal weights for all modes for simplicity
    log_probs_component <- -sq_dists / (2 * stdev^2)
    
    # combine modes using log-sum-exp
    # log(sum(exp(components)))
    log_prob_total <- torch_logsumexp(log_probs_component, dim = 2)
    
    # return Potential V(x) = -log(p(x))
    # we negate because MCMC minimizes Energy (High Prob = Low Energy)
    return(-log_prob_total)
  }
}

# --- Examples ----

# 3-Mode Triangle ---

# # A triangle layout: Top, Bottom-Left, Bottom-Right
# means_3 <- matrix(c(
#   0,  2.0,  # Top
#   -2, -1.5,  # Left
#   2, -1.5   # Right
# ), ncol = 2, byrow = TRUE)

# V_3modes <- make_gmm_density(means_3, stdev = 0.6)

# # --- 2-Mode Dumbbell ---
# # Two modes separated horizontally
# means_2 <- matrix(c(
#   -2.5, 0,
#   2.5, 0
# ), ncol = 2, byrow = TRUE)

# V_2modes <- make_gmm_density(means_2, stdev = .6)
