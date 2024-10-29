import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# 1. Define the kernel and initialize the GP
kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, n_restarts_optimizer=10)

# 2. Initialize an empty dataset
X_sampled = np.empty((0, 1))  # No initial points
y_sampled = np.empty((0, 1))  # No initial density reductions

# 3. Iteratively sample and update the GP
for _ in range(10):  # Assuming we want 10 points
    # Define the region of interest
    X = np.linspace(0, 10, 1000).reshape(-1, 1)
    
    # Predict GP mean and variance
    mu, sigma = gp.predict(X, return_std=True)
    
    # Sample a new point based on the current GP
    sample_index = np.argmax(mu)  # Or np.random.choice based on density
    x_new = X[sample_index].reshape(1, -1)
    y_new = np.array([[0]])  # Set low value to create a "repulsive" effect

    # Update the sampled points and observed values
    X_sampled = np.vstack([X_sampled, x_new])
    y_sampled = np.vstack([y_sampled, y_new])
    
    # Refit the GP to include the new pseudo-observation
    gp.fit(X_sampled, y_sampled)

# X_sampled now contains the sampled points with reduced density in nearby areas

if __name__ == "__main__":
    
    