library(torch)
library(R6)

library(ggplot2)
library(tidyr)
library(dplyr)
# library(checkmate) # Optional but good for validation


# This may be more specific to linear models, but can change it later
ParticlePosterior <- R6Class("ParticlePosterior",
  public = list(
    samples = NULL,
    param_names = NULL,

    initialize = function(particles_tensor, param_names=NULL){
      # convert tensor to R matrix
      # assign names to parameters 
      if (inherits(particles_tensor, "torch_tensor")) {
        self$samples <- as.matrix(particles_tensor$detach()$cpu())
      } else {
        self$samples <- as.matrix(particles_tensor)
      }
      
      n_params <- ncol(self$samples)
      
      #  add names to parameters
      if (is.null(param_names)) {
        self$param_names <- paste0("theta_", 1:n_params)
      } else {
        if(length(param_names) != n_params) stop("param_names length mismatch")
        self$param_names <- param_names
      }
      colnames(self$samples) <- self$param_names      
    },
    as_df = function(){
      .df <- data.frame(self$samples)
      colnames(.df) <- self$param_names
      return(.df)
    },
    summary = function(probs = c(0.05, 0.5, 0.95)) {
      # Returns Mean, SD, and Quantiles (Credible Intervals)
      stats <- data.frame(
        Parameter = self$param_names,
        Mean = self$mean(),
        SD = self$sd()
      )
      quants <- self$quantiles(probs=probs)
      stats <- cbind(stats, quants)
      return(stats)
    },
    mean = function(){
      return(colMeans(self$samples))
    },
    sd = function(){
      return(apply(self$samples, 2, sd))
    },
    var = function(){
      return(apply(self$samples, 2, var))
    },
    median = function(){
      return(apply(self$samples,2,FUN=median))
    },
    map = function(){
      # Maximum A Posteriori (MAP) Estimate
      map_vals <- apply(self$samples, 2, function(x) {
        d <- density(x)
        return(d$x[which.max(d$y)])
      })
      names(map_vals) <- self$param_names
      return(map_vals)
    },
    cov = function(){
      return(cov(self$samples))
    },
    quantiles = function(probs = c(0.05, 0.5, 0.95)){      
      return(t(apply(self$samples, 2, quantile, probs = probs)))
    },
    # credibility_interval(confidence=.95){},
    plot_marginals = function() {
      # Plot Marginal Posterior Distributions
      df_long <- as.data.frame(self$samples) %>%
        pivot_longer(everything(), names_to = "Parameter", values_to = "Value")
      
      ggplot(df_long, aes(x = Value, fill = Parameter)) +
        geom_density(alpha = 0.5) +
        facet_wrap(~Parameter, scales = "free") +
        theme_bw() +
        labs(title = "Marginal Densities", y = "Density") +
        theme(legend.position = "none")
    }, 
    # Posterior Predictive Distribution ---
      #   This is specific to Linear Regression logic. 
      #   Computes y_pred = X * theta^T for every particle.
    predict = function(X_new) {
        # X_new: Matrix (N_obs x D_params)
        # Samples: (N_particles x D_params)
        # Result: (N_obs x N_particles)
        # Each column is a possible regression line consistent with the data
        preds <- X_new %*% t(self$samples)
        return(preds)
      },
      
    # --- Bayesian R-Squared ---
    # Returns vector of R^2 values (one per particle)
    get_r2 = function(X, y_true) {
      # 1. Get predictions (N_obs x N_particles)
      y_pred <- self$predict(X)
      
      # 2. Residuals
      # Broadcast subtraction: (N_obs x N_particles) - (N_obs)
      resid <- sweep(y_pred, 1, y_true, "-")
      
      # 3. Variances per particle (column-wise)
      var_pred <- apply(y_pred, 2, var)
      var_resid <- apply(resid, 2, var)
      
      # 4. Bayesian R2 = Var(fit) / (Var(fit) + Var(resid))
      r2_dist <- var_pred / (var_pred + var_resid)
      
      return(r2_dist)
    },
    
    # --- Plot R-Squared Distribution ---
    plot_r2 = function(X, y_true) {
      r2_vals <- self$get_r2(X, y_true)
      df <- data.frame(R2 = r2_vals)
      
      ggplot(df, aes(x = R2)) +
        geom_histogram(bins = 30, fill = "steelblue", color = "white", alpha = 0.7) +
        geom_vline(aes(xintercept = median(R2)), color = "red", linetype = "dashed") +
        labs(title = "Bayesian R-squared Distribution",
             subtitle = paste("Median R2:", round(median(r2_vals), 3)),
             x = "R-squared", y = "Count") +
        theme_minimal()
    }
#     # particle diversity diagnostic
#     # Calculates mean pairwise euclidean distance between particles
#     # Low values indicate (possible) mode collapse.
#     check_diversity = function() {
#       # dist() computes distance matrix. 
#       # take the mean of the lower triangle.
#       d_mat <- dist(self$samples)
#       avg_dist <- mean(d_mat)
      
#       return(invisible(avg_dist))
#     },    
  )
)

# Examples ---- 

analysis <- ParticlePosterior$new(solver$get_final_particles())

# Summary
analysis$summary()

# MAP
analysis$map()

analysis$cov()

# Marginal Posterior Densities 
analysis$plot_marginals()

# Bayesian R^2
analysis$plot_r2(X_raw, y_raw)

