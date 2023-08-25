import numpy as np

# You are not allowed to use any ML libraries e.g. sklearn, scipy, keras, tensorflow etc

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py
# DO NOT INCLUDE OTHER PACKAGES LIKE SKLEARN, SCIPY, KERAS,TENSORFLOW ETC IN YOUR CODE
# THE USE OF ANY MACHINE LEARNING LIBRARIES WILL RESULT IN A STRAIGHT ZERO

# DO NOT CHANGE THE NAME OF THE METHOD my_fit BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

################################
# Non Editable Region Starting #
################################
def HT( v, k ):
      t = np.zeros_like( v )
      if k < 1:
          return t
      else:
          ind = np.argsort( abs( v ) )[ -k: ]
          t[ ind ] = v[ ind ]
          return t
def my_fit(X, y):
    S = 512
    w_init = np.linalg.lstsq(X, y, rcond=None)[0]

    learning_rate = 0.1
    max_iterations = 20
    tol = 1e-3
    n, D = X.shape

    model = w_init
    iteration = 0
    
    

    while iteration < max_iterations:
        error = X @ model - y
        gradient = X.T @ error / n
        updated_weights = model - learning_rate * gradient
        thresholded_weights = HT(updated_weights, S)

        non_zero_indices = np.nonzero(thresholded_weights)[0]
        X_non_zero = X[:, non_zero_indices]
        weights_non_zero = np.linalg.lstsq(X_non_zero, y, rcond=None)[0]

        thresholded_weights[non_zero_indices] = weights_non_zero

        if np.linalg.norm(thresholded_weights - model) < tol:
            break

        model = thresholded_weights
        iteration += 1
    return model

################################
#  Non Editable Region Ending  #
################################

	# Use this method to train your model using training CRPs
	# Youe method should return a 2048-dimensional vector that is 512-sparse
	# No bias term allowed -- return just a single 2048-dim vector as output
	# If the vector your return is not 512-sparse, it will be sparsified using hard-thresholding
						# Return the trained model

