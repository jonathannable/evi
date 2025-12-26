library(tidyverse)
library(brms)


source("./evi_solver.R")
source("./kernels.R")
source("./densities/toyobjects.R")
source("./plotting.R")

# Run a Logistic Regression on simulated data

# Setup Posterior 
LogisticRegressionPotential2 <- R6Class("LogisticRegressionPotential",
  inherit = PotentialBase,
  public = list(
    X = NULL,
    y = NULL,
    prior_var = NULL,
    initialize = function(X, y, alpha = 100.0) { # Default alpha=100 (Very weak prior)
      self$X <- if(is_torch_dtype(X)) X else torch_tensor(X, dtype=torch_float())
      self$y <- if(is_torch_dtype(y)) y else torch_tensor(y, dtype=torch_float())
      self$prior_var <- alpha
   },
   compute = function(particles) {
      # Prior ~ Normal(0,alphaI)
      # Prior: ||w||^2 / (2*alpha)
      prior_energy <- torch_sum(particles^2, dim = 2) / (2 * self$prior_var)
      # prior_energy <- 1
      # Likelihood
      logits <- torch_matmul(self$X, particles$t())
      y_broad <- self$y$unsqueeze(2)$expand_as(logits)
      # Combine Binary Cross Entropy with a sigmoid layer
      # Torch R says this is more stable... idk if it's necassary
      # Binary Cross Entropy loss is equivalent 
      # to the negative log-likelihood of a Bernoulli distribution
      # sigmoid(betax)^y(1-sigmoid(betax))^{1-y}
      loss_fn <- nn_bce_with_logits_loss(reduction = 'none')
      # neg log-lik
      nll_per_particle <- torch_sum(loss_fn(logits, y_broad), dim = 1)
      return(prior_energy + nll_per_particle)
   }
  )
)

set.seed(123)

N <- 1000
true_beta <- c(-1.5, 2.0, -0.8) 


X_raw <- matrix(rnorm(N * 2), ncol = 2)
colnames(X_raw) <- c("Var1", "Var2")
X <- cbind(Intercept = 1, X_raw)

# Calculate Probabilities
logits <- X %*% true_beta
probs  <- 1 / (1 + exp(-logits))

# Simulate Outcomes
y <- rbinom(N, 1, probs)



# important tuning for accuracy:
  # 1. alpha (prior var): set high (e.g., 50 or 100). 
  #    if alpha is small (e.g., 1), evi will shrink estimates toward 0 unlike glm.
  # 2. h (bandwidth): set small (e.g., 0.3 - 0.5).
  #    if h is large, particles push each other away too much, creating bias.

# EVI ----
lrtarget <- LogisticRegressionPotential2$new(X, y, alpha = 50)
x_init <- torch_randn(200, 3)

lrsolver <- EVISolver$new(
  target_neg_log_prob = lrtarget,
  kernel = GaussianKernel$new(h = 1), # Tighter kernel for better accuracy here
  temp = 1, 
  tau = 1e-2
)

lrsolver$fit(
  x_init, 
  time_steps = 100, 
  inner_steps = 50, 
  optimizer = optim_adam, 
  optimizer_config = list(lr = 5e-4),
  verbose = TRUE
)

# Get Means
evi_particles <- as.matrix(lrsolver$get_final_particles())
colnames(evi_particles) <- c("(Intercept)", "Var1", "Var2")
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

df_evi <- as.data.frame(evi_particles) %>%
  pivot_longer(cols = everything(), names_to = "Parameter", values_to = "Value")

df_evi$method <- "EVI"
# print(summary(evi_particles))

# GLM ----
glm_data <- data.frame(y = y, X_raw)
glm_model <- glm(y ~ Var1 + Var2, data = glm_data, family = binomial)
glm_coefs <- coef(glm_model)
glm_ci <- confint(glm_model)

# 95% CI
confint(glm_model)

df_glm <- data.frame(Parameter = names(glm_coefs), Value = as.numeric(glm_coefs))
df_glm$method <- "GLM"

# BRMS ----

priors <- c(
  prior(normal(0, 5), class = "Intercept"),
  prior(normal(0, 2.5), class = "b")
)
# No U-Turns Sampler (NUTS) MCMC
brms_model <- brm(
  formula = y ~ Var1 + Var2,
  data = glm_data,
  family = bernoulli(link = "logit"),
  prior = priors,
  chains = 4,
  iter = 4000,
  warmup = 1000,
  cores = 4,
  seed = 123
)
# summary(brms_model)
# 95% cred intervals
# posterior_interval(brms_model, prob = 0.95)
# posterior_means <- fixef(brms_model)[, "Estimate"]
brms_coefs <- fixef(brms_model)
post_draws <- as_draws_df(brms_model)



df_brms <- post_draws %>%
  select(b_Intercept, b_Var1, b_Var2) %>%
  pivot_longer(
    cols = everything(),
    names_to = "Parameter",
    values_to = "Value"
  ) %>%
  mutate(
    Parameter = case_match(
      Parameter,
      "b_Intercept" ~ "(Intercept)",
      "b_Var1" ~ "Var1",
      "b_Var2" ~ "Var2"
    ),
    method = "BRMS"
  )


# Table ----

results_table <- data.frame(
  Parameter = names(glm_coefs),
  Truth = true_beta,
  GLM_Estimate = as.numeric(glm_coefs),
  EVI_Mean = as.numeric(evi_means),
  ModelDiff = abs(as.numeric(glm_coefs) - as.numeric(evi_means)),
  TrueDiff = abs(as.numeric(true_beta) - as.numeric(evi_means))
)

print(results_table)

rownames(glm_ci) <- rownames(brms_coefs)
evi_coefs_df[,1] <- rownames(brms_coefs)
comparison <- data.frame(
  term = rownames(glm_ci),
  actuals = true_beta,
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


# For latex table
library(knitr)
kable(
  comparison_all %>% mutate(across(where(is.numeric), ~ round(.x, 3))),
  format = "latex",
  caption = "Logistic Regression Simulation Estimates",
  align = "c"
)
# Plot ----

df_truth <- data.frame(Parameter = names(glm_coefs), Value = true_beta)
df_truth$method <- "TRUTH"

ggplot() +
  # EVI Density
  geom_density(data = df_evi, aes(x = Value), fill="lightblue", alpha = 0.2, color="lightblue") +
  # BRMS Density
  geom_density(data = df_brms,aes(x = Value),fill="darkred",alpha = 0.1, color="darkred")+
  # True Value (Black Line)
  geom_vline(data = df_truth, aes(xintercept = Value), size = 1.2) +
  # GLM Value (Red Dashed)
  geom_vline(data = df_glm, aes(xintercept = Value), color = "red", linetype = "dashed", size = 1) +
  facet_wrap(~Parameter, scales = "free") +
  theme_bw() +
  labs(title = "Logistic Regression - Simulation",
       subtitle = "Black = Actual, Dashed = GLM, Blue = EVI, Red = MCMC",
       #caption = "If Red and Black are far from the Filled area, adjust 'alpha' (higher) or 'h' (lower)"
  )

