
# Base Class 
StopRule <- R6::R6Class("StopRule", public = list(
  check = function(state) stop("Must implement check()")
))


# Check gradient norm in inner optimization step
StopGradNorm <- R6::R6Class("StopGradNorm", inherit = StopRule, public = list(
  tol = NULL,
  initialize = function(tol = 1e-4) self$tol <- tol,
  check = function(state) {
    if (is.null(state$x$grad)) {
      return(FALSE)
    }
    return(state$x$grad$norm()$item() < self$tol)    
  }
))

# Check movement of particles has stopped 
StopDisplacement <- R6::R6Class("StopDisplacement", inherit = StopRule, public = list(
  tol = NULL,
  initialize = function(tol = 1e-5) {
    self$tol <- tol
  },
  check = function(state) {
    if (is.null(state$x_prev)) {
      return(FALSE)
    }
    # Mean Euclidean Distance between current state and previous 
    move <- torch_mean(torch_sqrt(torch_sum((state$x - state$x_prev)^2, dim = 2)))$item()
    return(move < self$tol)
  }
))

# Check energy has plateaued 
StopEnergyPlateau <- R6::R6Class("StopEnergyPlateau", inherit = StopRule, public = list(
  tol = NULL, window_size = NULL, history = c(),
  initialize = function(tol = 1e-6, window_size = 5) {
    self$tol <- tol
    self$window_size <- window_size
  },
  check = function(state) {
    self$history <- c(self$history, state$energy)
    
    if (length(self$history) < self$window_size){
      return(FALSE)
    }
    # get the last n items from the "tail" of the history
    self$history <- tail(self$history, self$window_size)
    return((max(self$history) - min(self$history)) < self$tol)
  }
))