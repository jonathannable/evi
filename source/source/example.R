library(torch)
library(R6)
library(tidyverse)
library(gridExtra)

source("./stop_rules.R")
source("./particles.R")
source("./evi_solver.R")
source("./densities/toyobjects.R")
source("./plotting.R")
 
# Define a Target ----
# -log(density) or "potential"
# A simple normal target
simple_target_func <- function(x) {
  # Optimization algorithms usually minimize a loss.
  # We want to maximize Probability.
  # Therefore, the function we pass to the solver must be the 
  # Negative Log Probability (also called Negative Log Likelihood or NLL).
  # Let Z ~ N(0,1), then 
  # -log(f_Z(z)) propto 1/2 sum(x^2)
  0.5 * torch_sum(x^2, dim = 2)
}

# 2. Pick the -log(density)/potential to use 
# Note: EVISolver can take an R6Object or a function like above
# We can also create a small wrapper function to call $compute()
# And evaluate the R6Objects directly
# target_func <- function(x) gmm_potential$compute(x)

means_triangle <- matrix(c(
  0,  2.0,   
  -2, -1.5,  
  2, -1.5    
), ncol = 2, byrow = TRUE)

target_func <- GMMPotential$new(means_triangle, stdev = 0.6)
# target_func <- StarPotential$new()
target_func <- BananaPotential$new()
# Setup Data ----
N <- 200 # Particles
D <- 2  # Parameters
# Standard Normal initial data
# We can offset
x_init <- torch_randn(N, D) # + 5 


# Define Kernel ----
gauss_kernel <- GaussianKernel$new(h = .25)
# lap_kernel <- LaplacianKernel$new(h = 1.0) 

# Config Solver ----
solver <- EVISolver$new(
  target_neg_log_prob = target_func,
  kernel = gauss_kernel,
  temp = 1.0,  # Higher temp = more spread out #I think this is not from the paper? where did I get this
  tau = 1e-2    # Small tau - slow accurate movements, high friction: Large tau - big jumps faster convergence
)

# Different Optimizers can be supplied
# Adam - Default
# solver$fit(x_init, 
#            optimizer = optim_adam, 
#            optimizer_config = list(lr = 0.05))

# SGD
# solver$fit(x_init, 
#            optimizer = optim_sgd, 
#            optimizer_config = list(lr = 0.1, momentum = 0.9))

# RMSProp 
# solver$fit(x_init, 
#            optimizer = optim_rmsprop, 
#            optimizer_config = list(lr = 0.01, alpha = 0.99))

# evi_solver is a convenience method to skip calling $new
solver <- evi_solver(
  target_neg_log_prob = target_func,
  kernel = gauss_kernel,
  temp = 1.0,  # Higher temp = more spread out #I think this is not from the paper? where did I get this
  tau = 1e-2    # Small tau - slow accurate movements, high friction: Large tau - big jumps faster convergence
)

# Define stop rules
outer_stop_rules <- list(StopDisplacement$new(1e-5), StopEnergyPlateau$new())
# Run ----
cat("Starting EVI Optimization...\n")
solver$fit(x_init, 
           time_steps = 2000, 
           inner_steps = 10,
           optimizer = optim_adam, 
           optimizer_config=list(lr=1e-3),
           outer_stop_rules=outer_stop_rules)

# Plots ----
# Basic plot for 2d data
final_x <- solver$get_final_particles()
plot(final_x, pch = 19, col = rgb(0,0,1,0.6), 
     main = "Final Particle Positions", 
     xlim = c(-3, 8), ylim = c(-3, 8))
points(as.matrix(x_init), col = "red", pch = 3) # Initial
legend("topright", legend=c("Initial", "Final"), col=c("red", "blue"), pch=c(3,19))

# Particle Trajectory Plots
plot_pbvi_evolution(
  trajectory = solver$trajectory, 
  target_fn = solver$target_neg_log_prob, # Can be passed directly!
  paths = TRUE
)

plot_pbvi_series(
  trajectory = solver$trajectory, 
  target_fn = solver$target_neg_log_prob
)

# These are KDE plots of the particles using ggplot
# Similar to the plot_pbvi_* functions above
ggplot(as.data.frame(final_x) ,aes(x=V1,y=V2))+geom_density_2d_filled()+geom_point()

last_step <- length(solver$trajectory)-1
trajectory_df <- lapply(c(1,10,50, last_step), function(i){
  as.data.frame(solver$trajectory[[i]]) %>% 
    setNames(c("x","y")) %>%
    mutate(
      label=paste("Time Step", i),
      step=i
    )
  }) %>% bind_rows() %>% arrange(step)

ggplot(trajectory_df, aes(x = x, y = y)) +
  geom_density_2d_filled(contour_var = "density") + 
  geom_point(color="lightblue", alpha = 0.1, size = 0.5)+
  facet_wrap(~ step, ncol = 4)+theme(legend.position = "none")

# not well tested
analysis <- solver$get_posterior()
analysis$summary()
