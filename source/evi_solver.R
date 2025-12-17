library(torch)
library(R6)


# 
# The EVI Solver - Implicit Euler Scheme 
# From Paper Particle-based Energetic Variational Inference by Kang et al.
# 

EVISolver <- R6Class("EVISolver",
  public = list(

    target_neg_log_prob = NULL, 
    kernel = NULL,
    temp = NULL,
    tau = NULL,
    trajectory = list(),
    
    # --- Constructor ----
    initialize = function(target_neg_log_prob, kernel, temp = 1.0, tau = 0.1) {
      # We expect a function/object that returns -log(p(x))
      self$target_neg_log_prob <- target_neg_log_prob
      self$kernel <- kernel
      self$temp <- temp
      self$tau <- tau
    },
    
    # --- Energy Function ----
    calc_energy = function(x) {
      # 1. Kernel / Entropy Term (Repulsive Force)
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
      
      # Target Probability Term - attractive force
      # Check if input is R6 class or standard function
      if (R6::is.R6(self$target_neg_log_prob)) {
        # Assumes R6 object has a 'compute' method
        target_val <- self$target_neg_log_prob$compute(x)
      } else {
        target_val <- self$target_neg_log_prob(x)
      }
      
      potential_term <- torch_mean(target_val)

      # Total Free Energy to Minimize
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
                   optimizer = optim_adam, 
                   optimizer_config = list(lr = 0.05),
                   verbose = TRUE) {
      
      x <- x_init$clone()$detach()$requires_grad_(TRUE)
      self$trajectory <- list(as.matrix(x$detach()$cpu()))
      
      for (t in 1:time_steps) {
        x_prev <- x$clone()$detach()
        
        # Initialize Optimizer dynamically
        optim_args <- c(list(params = list(x)), optimizer_config)
        opt <- do.call(optimizer, optim_args)
        
        for (k in 1:inner_steps) {
          opt$zero_grad()
          
          # Loss = Energy + Movement Cost
          loss <- self$calc_energy(x) + self$calc_jko_reg(x, x_prev)
          
          loss$backward()
          opt$step()
        }
        
        self$trajectory[[length(self$trajectory) + 1]] <- as.matrix(x$detach()$cpu())
        
        if (verbose && (t %% 5 == 0 || t == 1)) {
           cat(sprintf("Step %03d | Energy: %.4f\n", t, self$calc_energy(x)$item()))
        }
      }
      return(invisible(self))
    },
    
    get_final_particles = function() {
      if (length(self$trajectory) == 0) return(NULL)
      return(self$trajectory[[length(self$trajectory)]])
    },
    # Might remove or use more standard R type plots
    plot = function(type = "evolution", ...) {      
      if (type == "evolution") {
        if (!exists("plot_pbvi_evolution")) {
          stop("Required function 'plot_pbvi_evolution' is not loaded.")
        }        
        plot_pbvi_evolution(self$trajectory, self$target_neg_log_prob, ...)
      } else if (type == "series") {
        if (!exists("plot_pbvi_series")) {
          stop("Required function 'plot_pbvi_series' is not loaded.")
        }        
        plot_pbvi_series(self$trajectory, self$target_neg_log_prob, ...)
      }
    }    
  )
)