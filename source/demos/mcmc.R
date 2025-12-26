library(torch)

# Single Chain of MH
metropolis_hastings <- function(log_target, n_samples=5000, step_size=0.5, start_point=NULL) {
      
      # 1. Initialization
      if (is.null(start_point)) {
        # Start at random point (1, 2)
        x_curr <- torch_randn(c(1, 2)) 
      } else {
        x_curr <- start_point
      }
      
      # inital state
      v_curr <- log_target(x_curr)
      
      
      samples <- torch_zeros(c(n_samples, 2))
      accepted_count <- 0
      
      cat("Running Single-Chain MCMC...\n")
      
      for (t in 1:n_samples) {

        noise <- torch_randn(c(1, 2)) * step_size
        x_prop <- x_curr + noise      
        v_prop <- log_target(x_prop)
        log_ratio <- v_curr - v_prop      
        log_u <- torch_log(torch_rand(1))
                
        if (log_u$item() < log_ratio$item()) {
          # ACCEPT
          x_curr <- x_prop
          v_curr <- v_prop
          accepted_count <- accepted_count + 1
        } 
        # If REJECT, x_curr stays the same and  record the old position again
      
        samples[t, ] <- x_curr
      }
      # mixing statistic
      acc_rate <- accepted_count / n_samples
      cat(sprintf("Finished. Acceptance Rate: %.2f%%\n", acc_rate * 100))
      
      return(samples)
    }