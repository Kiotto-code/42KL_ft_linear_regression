import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define a simple quadratic function to optimize
# f(x, y) = x^2 + 4*y^2 + 2*x*y - 4*x - 8*y + 10
def objective_function(x, y):
    return x**2 + 4*y**2 + 2*x*y - 4*x - 8*y + 10

# Gradient (first derivatives)
def gradient(x, y):
    df_dx = 2*x + 2*y - 4
    df_dy = 8*y + 2*x - 8
    return np.array([df_dx, df_dy])

# Hessian matrix (second derivatives)
def hessian(x, y):
    # For this function, Hessian is constant
    d2f_dx2 = 2      # ∂²f/∂x²
    d2f_dy2 = 8      # ∂²f/∂y²
    d2f_dxdy = 2     # ∂²f/∂x∂y
    
    return np.array([[d2f_dx2, d2f_dxdy],
                     [d2f_dxdy, d2f_dy2]])

# Standard Gradient Descent
def gradient_descent(start_x, start_y, learning_rate=0.1, max_iterations=50):
    x, y = start_x, start_y
    path = [(x, y)]
    
    for i in range(max_iterations):
        grad = gradient(x, y)
        
        # Update: move in opposite direction of gradient
        x = x - learning_rate * grad[0]
        y = y - learning_rate * grad[1]
        
        path.append((x, y))
        
        # Stop if gradient is very small (converged)
        if np.linalg.norm(grad) < 1e-6:
            break
    
    return path

# Newton's Method (uses Hessian)
def newton_method(start_x, start_y, max_iterations=10):
    x, y = start_x, start_y
    path = [(x, y)]
    
    for i in range(max_iterations):
        grad = gradient(x, y)
        hess = hessian(x, y)
        
        # Newton's update: x_new = x - H^(-1) * gradient
        try:
            hess_inv = np.linalg.inv(hess)
            update = hess_inv @ grad
            
            x = x - update[0]
            y = y - update[1]
            
            path.append((x, y))
            
            # Stop if gradient is very small (converged)
            if np.linalg.norm(grad) < 1e-6:
                break
                
        except np.linalg.LinAlgError:
            print("Hessian is singular, cannot invert")
            break
    
    return path

# Example usage and comparison
if __name__ == "__main__":
    # Starting point
    start_x, start_y = 3.0, 2.0
    
    print(f"Starting point: ({start_x}, {start_y})")
    print(f"Starting function value: {objective_function(start_x, start_y):.4f}")
    
    # Run gradient descent
    gd_path = gradient_descent(start_x, start_y, learning_rate=0.1)
    print(f"\nGradient Descent:")
    print(f"Iterations: {len(gd_path) - 1}")
    print(f"Final point: ({gd_path[-1][0]:.4f}, {gd_path[-1][1]:.4f})")
    print(f"Final function value: {objective_function(gd_path[-1][0], gd_path[-1][1]):.4f}")
    
    # Run Newton's method
    newton_path = newton_method(start_x, start_y)
    print(f"\nNewton's Method (with Hessian):")
    print(f"Iterations: {len(newton_path) - 1}")
    print(f"Final point: ({newton_path[-1][0]:.4f}, {newton_path[-1][1]:.4f})")
    print(f"Final function value: {objective_function(newton_path[-1][0], newton_path[-1][1]):.4f}")
    
    # The true minimum (can be found analytically)
    print(f"\nTrue minimum is at (0, 1) with value {objective_function(0, 1):.4f}")
    
    # Plot the comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Create a grid for contour plot
    x_range = np.linspace(-1, 4, 100)
    y_range = np.linspace(-1, 3, 100)
    X, Y = np.meshgrid(x_range, y_range)
    Z = objective_function(X, Y)
    
    # Plot 1: Gradient Descent
    ax1.contour(X, Y, Z, levels=20, alpha=0.7)
    gd_x, gd_y = zip(*gd_path)
    ax1.plot(gd_x, gd_y, 'ro-', linewidth=2, markersize=4, label='Gradient Descent')
    ax1.plot(gd_x[0], gd_y[0], 'go', markersize=8, label='Start')
    ax1.plot(gd_x[-1], gd_y[-1], 'bs', markersize=8, label='End')
    ax1.plot(0, 1, 'k*', markersize=10, label='True Minimum')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Gradient Descent Path')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Newton's Method
    ax2.contour(X, Y, Z, levels=20, alpha=0.7)
    newton_x, newton_y = zip(*newton_path)
    ax2.plot(newton_x, newton_y, 'mo-', linewidth=2, markersize=4, label="Newton's Method")
    ax2.plot(newton_x[0], newton_y[0], 'go', markersize=8, label='Start')
    ax2.plot(newton_x[-1], newton_y[-1], 'bs', markersize=8, label='End')
    ax2.plot(0, 1, 'k*', markersize=10, label='True Minimum')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title("Newton's Method Path")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Show step-by-step comparison
    print("\nStep-by-step comparison:")
    print("Step | Gradient Descent      | Newton's Method")
    print("-----|----------------------|----------------")
    max_steps = max(len(gd_path), len(newton_path))
    
    for i in range(min(10, max_steps)):  # Show first 10 steps
        gd_str = f"({gd_path[i][0]:.2f}, {gd_path[i][1]:.2f})" if i < len(gd_path) else "Done"
        newton_str = f"({newton_path[i][0]:.2f}, {newton_path[i][1]:.2f})" if i < len(newton_path) else "Done"
        print(f"{i:4d} | {gd_str:20s} | {newton_str}")

# Simple machine learning example: Linear regression with gradient descent
print("\n" + "="*60)
print("BONUS: Simple Linear Regression Example")
print("="*60)

# Generate some sample data
np.random.seed(42)
X = np.random.randn(100, 1)
y = 3 * X.flatten() + 2 + 0.5 * np.random.randn(100)  # y = 3x + 2 + noise

# Add bias term (intercept)
X_with_bias = np.column_stack([np.ones(len(X)), X])

def mean_squared_error(weights, X, y):
    """Calculate mean squared error"""
    predictions = X @ weights
    return np.mean((predictions - y) ** 2)

def mse_gradient(weights, X, y):
    """Calculate gradient of MSE"""
    predictions = X @ weights
    return 2 * X.T @ (predictions - y) / len(y)

# Gradient descent for linear regression
def linear_regression_gd(X, y, learning_rate=0.01, max_iterations=1000):
    weights = np.random.randn(X.shape[1])  # Random initialization
    costs = []
    
    for i in range(max_iterations):
        cost = mean_squared_error(weights, X, y)
        grad = mse_gradient(weights, X, y)
        
        weights = weights - learning_rate * grad
        costs.append(cost)
        
        if i % 100 == 0:
            print(f"Iteration {i}, Cost: {cost:.4f}")
    
    return weights, costs

# Run linear regression
print("Training linear regression with gradient descent...")
final_weights, costs = linear_regression_gd(X_with_bias, y, learning_rate=0.1)

print(f"\nFinal weights: Intercept = {final_weights[0]:.4f}, Slope = {final_weights[1]:.4f}")
print(f"True values:   Intercept = 2.0000, Slope = 3.0000")
print(f"Final cost: {costs[-1]:.4f}")

# Plot the results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Data and fitted line
ax1.scatter(X, y, alpha=0.6, label='Data')
x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
x_line_with_bias = np.column_stack([np.ones(len(x_line)), x_line])
y_line = x_line_with_bias @ final_weights
ax1.plot(x_line, y_line, 'r-', linewidth=2, label='Fitted Line')
ax1.set_xlabel('X')
ax1.set_ylabel('y')
ax1.set_title('Linear Regression Result')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Cost function over iterations
ax2.plot(costs)
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Mean Squared Error')
ax2.set_title('Cost Function During Training')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()