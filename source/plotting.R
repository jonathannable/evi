library(ggplot2)
library(gridExtra)


# Helpers  ----

# Turn potential/target/negative log-likelihood into (unnormalized) probability for plotting
get_prob_fn_from_potential <- function(potential_source) {
  
  # Check if the input is an R6 object from this library or a regular R function
  is_r6 <- R6::is.R6(potential_source)
  
  function(x) {
    # get the energy/potential values
    if (is_r6) {
      v_val <- potential_source$compute(x)
    } else {
      v_val <- potential_source(x)
    }
    
    # convert to probability: p(x) ~ exp(-V(x))
    # detach  and convert to array for plotting logic
    # detach removes it from the autograd graph
    v_arr <- as_array(v_val$detach())
    
    # subtract min for numerical stability in visualization 
    # so the brightest spot is always 1.0
    min_v <- min(v_arr)
    exp(-(v_arr - min_v))
  }
}

get_density_grid <- function(density_fn, xlim, ylim, res = 100) {
  grid_x <- seq(xlim[1], xlim[2], length.out = res)
  grid_y <- seq(ylim[1], ylim[2], length.out = res)
  grid_df <- expand.grid(x = grid_x, y = grid_y)
  
  grid_tensor <- torch_tensor(as.matrix(grid_df), dtype = torch_float())
  
  with_no_grad({
    density_vals <- density_fn(grid_tensor)
  })
  
  grid_df$z <- as.numeric(density_vals)
  return(grid_df)
}

get_trajectory_limits <- function(trajectory, padding = 0.5) {
  # The solver now stores matrices, but we keep the check just in case
  all_points <- do.call(rbind, lapply(trajectory, function(x) {
    if(inherits(x, "torch_tensor")) as_array(x$detach()) else x
  }))
  
  x_range <- range(all_points[,1])
  y_range <- range(all_points[,2])
  
  return(list(
    xlim = x_range + c(-padding, padding), 
    ylim = y_range + c(-padding, padding)
  ))
}

# Plots ----

# EVI Plots ----

plot_pbvi_series <- function(trajectory, target_fn, steps_to_plot = NULL, 
                             xlim = NULL, ylim = NULL) {
  
  if (is.null(xlim) || is.null(ylim)) {
    limits <- get_trajectory_limits(trajectory)
    if (is.null(xlim)) xlim <- limits$xlim
    if (is.null(ylim)) ylim <- limits$ylim
  }
  
  # Use updated helper that handles R6 objects
  prob_fn <- get_prob_fn_from_potential(target_fn)
  grid_data <- get_density_grid(prob_fn, xlim, ylim)
  
  if (is.null(steps_to_plot)) {
    steps_to_plot <- seq(1, length(trajectory), length.out = 6) %>% round()
    steps_to_plot[1] <- 1
  }
  
  plot_list <- list()
  
  for (step_idx in steps_to_plot) {
    step_data <- trajectory[[step_idx]]
    if (inherits(step_data, "torch_tensor")) step_data <- as_array(step_data$detach())
    
    particles <- as.data.frame(step_data)
    colnames(particles) <- c("px", "py")
    
    p <- ggplot() +
      geom_contour_filled(data = grid_data, aes(x = x, y = y, z = z), 
                          bins = 10, alpha = 0.6, show.legend = FALSE) +
      scale_fill_viridis_d(option = "D") +
      
      geom_point(data = particles, aes(x = px, y = py), 
                 fill = "red", color = "white", pch = 21, size = 1.5, alpha = 0.8) +
      
      coord_fixed(xlim = xlim, ylim = ylim) + # FIXED MISSING PLUS HERE
      theme_minimal() +
      theme(axis.text = element_blank(), axis.ticks = element_blank(),
            panel.grid = element_blank()) +
      ggtitle(paste("Step:", step_idx))
    
    plot_list[[length(plot_list) + 1]] <- p
  }
  
  do.call(gridExtra::grid.arrange, c(plot_list, ncol = 3))
}

plot_pbvi_evolution <- function(trajectory, target_fn, paths=TRUE, max_lines = 30, 
                                xlim = NULL, ylim = NULL) {
  
  if (is.null(xlim) || is.null(ylim)) {
    limits <- get_trajectory_limits(trajectory)
    if (is.null(xlim)) xlim <- limits$xlim
    if (is.null(ylim)) ylim <- limits$ylim
  }
  
  prob_fn <- get_prob_fn_from_potential(target_fn)
  grid_data <- get_density_grid(prob_fn, xlim, ylim)
  
  # Data Prep
  n_particles <- if(inherits(trajectory[[1]], "torch_tensor")) nrow(trajectory[[1]]) else nrow(trajectory[[1]])
  
  if (n_particles > max_lines) {
    indices_to_plot <- sample(1:n_particles, max_lines)
  } else {
    indices_to_plot <- 1:n_particles
  }
  
  traj_list <- list()
  for (t in seq_along(trajectory)) {
    mat <- trajectory[[t]]
    if (inherits(mat, "torch_tensor")) mat <- as_array(mat$detach())
    
    mat_sub <- mat[indices_to_plot, , drop=FALSE]
    df <- as.data.frame(mat_sub)
    colnames(df) <- c("px", "py")
    df$step <- t
    df$particle_id <- indices_to_plot
    traj_list[[t]] <- df
  }
  
  full_traj_df <- do.call(rbind, traj_list)
  
  p <- ggplot() +    
    geom_contour(data = grid_data, aes(x = x, y = y, z = z), 
                 colour = "grey50", alpha = 0.7)
  
  if(paths){
    p <- p + geom_path(data = full_traj_df, 
                  aes(x = px, y = py, group = particle_id, color = step), 
                  alpha = 0.8, size = 0.6) +
      geom_point(data = subset(full_traj_df, step == 1), 
                 aes(x = px, y = py), color = "black", size = 1, shape = 3) 
  }
  
  p <- p + geom_point(data = subset(full_traj_df, step == length(trajectory)), 
                 aes(x = px, y = py), color = "red", size = 1.5) +
    scale_color_viridis_c(name = "Step", option = "magma") +
    coord_fixed(xlim = xlim, ylim = ylim) +
    theme_bw() + 
    theme(legend.position = "none")
  
  print(p)
}

# older functions, the plot_pbvi_* are updated but I think something still calls these...
plot_series <- function(trajectory, density_fn, steps_to_plot = NULL, 
                        xlim = NULL, ylim = NULL) {
  
  # 1. Determine Limits
  if (is.null(xlim) || is.null(ylim)) {
    limits <- get_trajectory_limits(trajectory)
    if (is.null(xlim)) xlim <- limits$xlim
    if (is.null(ylim)) ylim <- limits$ylim
  }
  
  # 2. Get Background Density
  grid_data <- get_density_grid(density_fn, xlim, ylim)
  
  # 3. Determine Steps
  if (is.null(steps_to_plot)) {
    steps_to_plot <- seq(1, length(trajectory), length.out = 6) %>% round()
  }
  
  plot_list <- list()
  
  for (step_idx in steps_to_plot) {
    # Get particles
    particles <- as.data.frame(trajectory[[step_idx]])
    colnames(particles) <- c("px", "py")
    
    p <- ggplot() +
      # Plot Density
      geom_contour_filled(data = grid_data, aes(x = x, y = y, z = z), 
                          bins = 10, alpha = 0.6, show.legend = FALSE) +
      scale_fill_viridis_d() +
      # Plot Particles
      geom_point(data = particles, aes(x = px, y = py), 
                 color = "red", size = 1, alpha = 0.7) +
      # Apply Limits
      coord_fixed(xlim = xlim, ylim = ylim) +
      labs(title = paste("Step", step_idx - 1), x = "", y = "") +
      theme_bw() +
      theme(axis.text = element_blank(), axis.ticks = element_blank())
    
    plot_list[[length(plot_list) + 1]] <- p
  }
  
  # Arrange
  do.call(gridExtra::grid.arrange, c(plot_list, ncol = length(steps_to_plot)))
}

plot_final_evolution <- function(trajectory, density_fn, steps_to_plot = NULL,
                                 xlim = NULL, ylim = NULL) {
  
  # 1. Determine Limits
  if (is.null(xlim) || is.null(ylim)) {
    limits <- get_trajectory_limits(trajectory)
    if (is.null(xlim)) xlim <- limits$xlim
    if (is.null(ylim)) ylim <- limits$ylim
  }
  
  # 2. Get Background Density
  grid_data <- get_density_grid(density_fn, xlim, ylim)
  
  if (is.null(steps_to_plot)) {
    steps_to_plot <- seq(1, length(trajectory), length.out = 5) %>% round()
  }
  
  # Base plot with density contours (lines only, so we can see points)
  p <- ggplot() +
    geom_contour(data = grid_data, aes(x = x, y = y, z = z), 
                 colour = "grey", alpha = 0.5) +
    coord_fixed(xlim = xlim, ylim = ylim) +
    theme_bw()
  
  # Add scatter points
  for (i in seq_along(steps_to_plot)) {
    step <- steps_to_plot[i]
    df_step <- as.data.frame(trajectory[[step]]) 
    colnames(df_step) <- c("px", "py")
    
    p <- p + geom_point(data = df_step, aes(x = px, y = py, color = as.factor(step - 1)), 
                        alpha = 0.6)
  }
  
  p <- p + scale_color_viridis_d(name = "Step")
  print(p)
}

# MCMC ----

# Helper to ensure MCMC chain is an R matrix
prepare_chain_data <- function(chain_tensor) {
  if (inherits(chain_tensor, "torch_tensor")) {
    return(as_array(chain_tensor$detach()))
  }
  return(chain_tensor)
}

plot_mcmc_series <- function(chain_tensor, target_fn, snapshots = NULL, 
                             xlim = NULL, ylim = NULL) {
  
  chain <- prepare_chain_data(chain_tensor)
  
  # Determine Limits
  if (is.null(xlim) || is.null(ylim)) {
    # Pad limits based on the visited area
    xlim <- range(chain[,1]) + c(-0.5, 0.5)
    ylim <- range(chain[,2]) + c(-0.5, 0.5)
  }
  
  # Get Background Density (Convert Potential to Prob Density)
  # We reuse your existing get_density_grid function
  prob_fn <- get_prob_fn_from_potential(target_fn)
  grid_data <- get_density_grid(prob_fn, xlim, ylim)
  
  # 3. Determine Snapshots (points in time to plot)
  n_samples <- nrow(chain)
  if (is.null(snapshots)) {
    # Create 6 evenly spaced snapshots
    snapshots <- seq(10, n_samples, length.out = 6) %>% round()
  }
  
  plot_list <- list()
  
  for (iter in snapshots) {
    # Get history up to this iteration
    chain_subset <- as.data.frame(chain[1:iter, , drop = FALSE])
    colnames(chain_subset) <- c("px", "py")
    
    p <- ggplot() +
      # Contour Background
      geom_contour_filled(data = grid_data, aes(x = x, y = y, z = z), 
                          bins = 10, alpha = 0.6, show.legend = FALSE) +
      scale_fill_viridis_d() +
      
      # Plot the Path 
      geom_path(data = chain_subset, aes(x = px, y = py), 
                color = "white", alpha = 0.3, size = 0.3) +
      
      # Plot the Points
      geom_point(data = chain_subset, aes(x = px, y = py), 
                 color = "red", size = 0.5, alpha = 0.6) +
      
      # Highlight the current head of the chain
      geom_point(data = chain_subset[iter, ], aes(x = px, y = py),
                 color = "yellow", size = 2, shape = 21, fill = "red") +
      
      coord_fixed(xlim = xlim, ylim = ylim) +
      labs(title = paste("Iter:", iter), x = "", y = "") +
      theme_minimal() +
      theme(axis.text = element_blank(), axis.ticks = element_blank())
    
    plot_list[[length(plot_list) + 1]] <- p
  }
  
  do.call(gridExtra::grid.arrange, c(plot_list, ncol = 3))
}

plot_mcmc_path <- function(chain_tensor, target_fn, xlim = NULL, ylim = NULL) {
  
  chain <- prepare_chain_data(chain_tensor)
  
  if (is.null(xlim)) {
    xlim <- range(chain[,1]) + c(-0.5, 0.5)
  }
  if(is.null(ylim)) {
    ylim <- range(chain[,2]) + c(-0.5, 0.5)
  }
  
  # Get Density Background
  prob_fn <- get_prob_fn_from_potential(target_fn)
  grid_data <- get_density_grid(prob_fn, xlim, ylim)
  
  # Prepare dataframe with time column
  df_chain <- as.data.frame(chain)
  colnames(df_chain) <- c("px", "py")
  df_chain$time <- 1:nrow(df_chain)
  
  p <- ggplot() +
    # Contours (Lines only for cleaner look)
    geom_contour(data = grid_data, aes(x = x, y = y, z = z), 
                 colour = "grey50", alpha = 0.7) +
    
    # Path colored by time
    geom_path(data = df_chain, aes(x = px, y = py, color = time), 
              alpha = 0.8, size = 0.5) +
    
    # Start Point
    geom_point(data = df_chain[1,], aes(x = px, y = py), shape = "S", size = 3) +
    
    # End Point
    geom_point(data = df_chain[nrow(df_chain),], aes(x = px, y = py), shape = "E", size = 3) +
    
    scale_color_viridis_c(name = "Iteration", option = "magma") +
    coord_fixed(xlim = xlim, ylim = ylim)+
    theme_bw()
  
  print(p+theme(legend.position = "none")+guides(color = "none"))
}
