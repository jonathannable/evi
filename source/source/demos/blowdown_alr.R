library(alr4)
library(tidyverse)
library(brms)

source("./evi_solver.R")
source("./kernels.R")
source("./plotting.R")

# The Blowdown model was taken from Chapter 12 of ALR by Sanford Weisberg.
# See "Blowdown" Chapter 12.2 Applied Linear Regression, page 274 for more information

# Posterior setup -----

LogisticRegressionPotential <- R6Class("LogisticRegressionPotential",
  inherit = PotentialBase,
  public = list(
    X = NULL,
    y = NULL,
    prior_var = NULL,
    
    initialize = function(X, y, alpha = 10.0) {
      # X: (N, D) matrix including intercept
      # y: (N) vector 0/1
      # alpha: Prior variance for weights (Higher = relaxed prior, closer to MLE)
      
      self$X <- if(is_torch_dtype(X)) X else torch_tensor(X, dtype=torch_float())
      self$y <- if(is_torch_dtype(y)) y else torch_tensor(y, dtype=torch_float())
      self$prior_var <- alpha
    },
    
    compute = function(particles) {
      # particles shape: (M_particles, D_dims)
      
      prior_energy <- torch_sum(particles^2, dim = 2) / (2 * self$prior_var)

      logits <- torch_matmul(self$X, particles$t())
      
      # Expand y to match logits (N, M)
      y_broad <- self$y$unsqueeze(2)$expand_as(logits)
      
      # binary cross entropy loss - aka bernoulli likelihood
      loss_fn <- nn_bce_with_logits_loss(reduction = 'none')
      nll_per_particle <- torch_sum(loss_fn(logits, y_broad), dim = 1)
      
      return(prior_energy + nll_per_particle)
    }
  )
)

# Blowdown Data ----
data(Blowdown)

df_blowdown <- Blowdown %>%
  filter(spp == "black spruce") %>%
  mutate(
    # Ensure y is 0/1 numeric
    y = as.numeric(y), 
    log_d = log(d),
    `log_d:s` = log_d*s,
  ) %>%
  select(y, log_d, s,`log_d:s`) %>%
  na.omit()


# EVI relies on calculating distances between particles.
# If one weight is for a variable with range (0, 0.01) and another (0, 1000),
# the Euclidean distance is distorted - i.e. we need to scale inputs.
X_raw <- df_blowdown %>% select(log_d, s, `log_d:s`)
X_scaled <- scale(X_raw) 

# Add Intercept Column
X_final <- cbind(Intercept = 1, X_scaled)
y_final <- df_blowdown$y

# Save feature names for plotting later
feature_names <- colnames(X_final)

cat(sprintf("Data Loaded: %d rows, %d predictors.\n", nrow(X_final), ncol(X_final)))



# EVI ----

# Target density
# Use a high prior variance (alpha=10) to mimic a flatish prior,
# allowing the data to dominate the posterior.
logit_target <- LogisticRegressionPotential$new(X_final, y_final, alpha = 50.0)


solver <- EVISolver$new(
  target_neg_log_prob = logit_target,
  kernel = GaussianKernel$new(h = 1.0), 
  temp = 1,   # Temperature controls spread (Uncertainty)
  tau = 1e-2    # Step size
)

# Initialize Particles
n_particles <- 200
n_dims <- ncol(X_final)
x_init <- torch_randn(n_particles, n_dims)


solver$fit(
  x_init, 
  time_steps = 100,
  inner_steps = 50, 
  optimizer = optim_adam, 
  optimizer_config = list(lr = 5e-4),
  verbose = TRUE
)


evi_particles <- as.matrix(solver$get_final_particles())
colnames(evi_particles) <- feature_names
evi_means <- colMeans(evi_particles)

# 95%
evi_95_int <- apply(evi_particles, 2, quantile, probs = c(0.025, 0.975))
evi_coefs <- rbind(evi_means, evi_95_int)
evi_coefs_df <- data.frame(
  term = colnames(evi_coefs),
  evi_est = evi_coefs["evi_means", ],
  evi_low = evi_coefs["2.5%", ],
  evi_high = evi_coefs["97.5%", ],
  row.names = NULL
)
# Convert to long format for ggplot
df_evi <- as.data.frame(evi_particles) %>%
  pivot_longer(cols = everything(), names_to = "Predictor", values_to = "Value")

# df_evi$Predictor[df_evi$Predictor == "log_d_s"] <- "log_d:s"

# BRMS ----

priors <- c(
  prior(normal(0, 2.5), class = "b"),
  prior(normal(0, 5), class = "Intercept")
)

brms_model <- brm(
  y ~ log_d + s + s:log_d,
  data = df_blowdown,
  family = bernoulli(link = "logit"),
  prior = priors,
  chains = 4,
  cores = 4,
  iter = 4000,
  warmup = 1000,
  seed = 123
)
# summary(brms_model)
# 95% cred intervals
brms_coefs <- fixef(brms_model)

# posterior_interval(brms_model, prob = 0.95)
# posterior_means <- fixef(brms_model)[, "Estimate"]

post_draws <- as_draws_df(brms_model)

df_brms <- post_draws %>%
  select(b_Intercept, b_log_d, b_s, `b_log_d:s`) %>%
  pivot_longer(
    cols = everything(),
    names_to = "Parameter",
    values_to = "Value"
  ) 
df_brms <- df_brms %>%
  mutate(
    Parameter = case_match(
      Parameter,
      "b_Intercept" ~ "(Intercept)",
      "b_log_d" ~ "log_d",
      "b_s" ~ "s",
      "b_log_d:s" ~ "log_d:s"      
    ),
    method = "BRMS"
  )

# GLM ----

glm_model <- glm(y ~ log_d + s + s:log_d, data = df_blowdown, family = binomial)
glm_coefs <- coef(glm_model)
glm_ci <- confint(glm_model)

df_glm <- data.frame(
  Predictor = names(glm_coefs),
  Mean = as.numeric(glm_coefs),
  Lower = glm_ci[,1],
  Upper = glm_ci[,2]
)

# Rename predictors in GLM to match X_final column names if needed
df_glm$Predictor[df_glm$Predictor == "(Intercept)"] <- "Intercept"



# Table ----

names(glm_coefs) <- feature_names
rownames(brms_coefs) <- feature_names

comparison <- data.frame(
  # term = rownames(glm_ci),
  term = feature_names,
  glm_est = glm_coefs,
  glm_low = glm_ci[,1],
  glm_high = glm_ci[,2],
  brms_est = brms_coefs[, "Estimate"],
  brms_low = brms_coefs[, "Q2.5"],
  brms_high = brms_coefs[, "Q97.5"]
)
comparison_all <- merge(
  comparison,
  evi_coefs_df,
  by = "term",
  all = TRUE
)
print(comparison_all)

# For latex table
library(knitr)
kable(
  comparison_all %>% mutate(across(where(is.numeric), ~ round(.x, 3))),
  format = "latex",
  caption = "Logistic Regression Simulation Estimates",
  align = "c"
)

# Plot ----

p <- ggplot() +
  # EVI Density
  geom_density(data = df_evi, aes(x = Value), fill="lightblue", alpha = 0.2, color="lightblue") +
  geom_density(data = df_brms,aes(x = Value),fill="darkred",alpha = 0.1, color="darkred")+
  # 2. Add GLM Estimates (Point estimate + vertical lines for CI)
  geom_vline(data = df_glm, aes(xintercept = Mean), linetype="dashed") +
  # geom_rect(data = df_glm, aes(xmin=Lower, xmax=Upper, ymin=0, ymax=Inf, fill=Predictor), 
  #           alpha = 0.1, inherit.aes = FALSE) +
  facet_wrap(~Predictor, scales = "free") +
  theme_bw() +
  labs(title = "ALR - Blowdown",
       subtitle = "Dashed = GLM, Blue = EVI, Red = MCMC",
       y = "Posterior Density", x = "Coefficient Value") +
  theme(legend.position = "none")

print(p)


