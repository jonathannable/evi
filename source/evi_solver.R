library(torch)
library(R6)


# 
# The EVI Solver - Implicit Euler Scheme 
# From Paper Particle-based Energetic Variational Inference by Kang et al.
# 

source("./stop_rules.R")
source("./particles.R")

EVISolver <- R6Class("EVISolver",
  public = list(

    target_neg_log_prob = NULL, 
    kernel = NULL,
    temp = NULL,
    tau = NULL,
    trajectory = list(),
    
    # --- Constructor ----
    initialize = function(target_neg_log_prob, kernel, temp = 1.0, tau = 0.1) {
      # expect a function/object that returns -log(p(x))
      self$target_neg_log_prob <- target_neg_log_prob
      # see kernels.R
      self$kernel <- kernel
      self$temp <- temp
      self$tau <- tau # Small tau - slow accurate movements, high friction: Large tau - big jumps faster convergence
    },
    
    # --- Energy Function ----
    calc_energy = function(x) {
      # kernel / entropy term: the repulsive force
      xi <- x$unsqueeze(2)
      xj <- x$unsqueeze(1)
      K_mat <- self$kernel$compute(xi, xj)
      
      epsilon <- 1e-10
      avg_density <- torch_mean(K_mat, dim = 2) + epsilon
      
      # 
      # Minimize: log(density). 
      # Turning up the temp causes particles to wiggle more
      # Minimizing a positive log-density pushes density down -> spreads particles.
      entropy_term <- self$temp * torch_mean(torch_log(avg_density))
      
      # Target probability term - attractive force
      # Check if input is R6 class or standard function
      if (R6::is.R6(self$target_neg_log_prob)) {
        # Assumes R6 object has a 'compute' method
        target_val <- self$target_neg_log_prob$compute(x)
      } else {
        target_val <- self$target_neg_log_prob(x)
      }
      
      potential_term <- torch_mean(target_val)

      # Total free energy to minimize
      return(entropy_term + potential_term)
    },
    
    # --- Regularization/Movement Cost ----
    calc_jko_reg = function(x, x_prev) {
      diff_sq <- torch_sum((x - x_prev)^2, dim = 2)
      return((1 / (2 * self$tau)) * torch_mean(diff_sq))
    },
    
    # --- Fit Loop -----
    fit = function(x_init, 
                   time_steps = 50, 
                   inner_steps = 100,
                   # tol_outer = 1e-5, # outter movement displacement tolerance
                   tol_inner = 1e-6, # inner optimizatin gradient tolerance 
                   outer_stop_rules = list(),
                   #inner_stop_rules = list(),                   
                   optimizer = optim_adam, 
                   optimizer_config = list(lr = 0.05),
                   verbose = TRUE) {
      
      x <- x_init$clone()$detach()$requires_grad_(TRUE)
      self$trajectory <- list(as.matrix(x$detach()$cpu()))
      tryCatch({
      for (t in 1:time_steps) {
        x_prev <- x$clone()$detach()
        
        # initialize optimizer
        optim_args <- c(list(params = list(x)), optimizer_config)
        opt <- do.call(optimizer, optim_args)
        
        for (k in 1:inner_steps) {
          opt$zero_grad()
          
          # Loss = Energy + MovementCost
          loss <- self$calc_energy(x) + self$calc_jko_reg(x, x_prev)
          
          loss$backward()

          # Stopping condition for the optimizer
          gnorm <- x$grad$norm()$item()
          if (gnorm < tol_inner){
            break
            if(verbose){
              cat(sprintf("Inner loop stopping condition met after %d steps \n", k))
            }
          }

          opt$step()
        }
        # Determine if the particles are still moving 
        # Small time steps may cause this to stop early
        # Uses mean euclidean distance between x_current and x_prev
        movement <- torch_mean(torch_sqrt(torch_sum((x - x_prev)^2, dim = 2)))$item()
        current_energy <- self$calc_energy(x)$item()
        self$trajectory[[length(self$trajectory) + 1]] <- as.matrix(x$detach()$cpu())
        
        if (verbose && (t %% 5 == 0 || t == 1)) {
           cat(sprintf("Step %03d | Energy: %.4f | Movement: %.4f\n", t, current_energy,movement))
        }

        if(length(outer_stop_rules) > 0){
          state_outer <- list(x = x, x_prev = x_prev, energy = current_energy, step = t)
          if(any(sapply(outer_stop_rules, function(r) r$check(state_outer)))){
            if (verbose){
                cat("Outer loop convergence reached.\n")
            }
            break          
          }
        }

        # if (movement < tol_outer) {
        #   if (verbose) cat("Converged: Movement below threshold.\n")
        #   break
        # }        
      }
      }, interrupt = function(i) {
            message("\nSolver interrupted.")
            message(sprintf("Stopping at time step %d. Current trajectory is saved in the solver object.", length(self$trajectory) - 1))
      }, error = function(e) {      
        message("\nAn error occurred during optimization:")
        print(e)
      })
      return(invisible(self))
    },
    
    get_final_particles = function() {
      if (length(self$trajectory) == 0) return(NULL)
      return(self$trajectory[[length(self$trajectory)]])
    },    
    get_posterior = function(param_names = NULL) {
      final_particles <- self$get_final_particles()
      if (is.null(final_particles)) {
        return(NULL)      
      }
      return(ParticlePosterior$new(final_particles, param_names))
    }  
  )
)

# factory function, because who likes to call "new"
# Including required bits for Rstudio to pick up on autocomplete
evi_solver <- function(target_neg_log_prob, kernel, ...) {
  EVISolver$new(target_neg_log_prob, kernel, ...)
}