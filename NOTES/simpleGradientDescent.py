import numpy as np
import matplotlib.pyplot as plt
import os
import csv
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, "car_data.csv")

def LoadFile():
    try:
        x = []
        y = []
        print("csv_path: ", csv_path)
        # input("csv_path continue 2")
        with open(csv_path, 'r') as file:
            model_data = csv.DictReader(file)
            print("Open file successfully")
            # print(list(model_data))
            input("Loadfile continue 1")
            for r in model_data:
                # print("r: ", r)
                # input("Loadfile continue 2")
                # _modelData.append(r)
                x.append(float(r['mileage']))  # Mileage is the km traveled by vehicles
                y.append(float(r['price'])) # Price is the Price of car after traveled Mileage
        # print (_modelData)
        print("LoadFile x : ", x)
        print("LoadFile y : ", y)
        input("Loadfile continue 3")
        return np.array(x), np.array(y)
    except Exception as e:
        # An unopenable file
        print(f"Error loading file: {e}")
        return None, None

def normalize(arr):
    mean = np.mean(arr)
    std = np.std(arr)
    return (arr - mean) / std, mean, std


def draw_plt (plot, x_label, y_label, title, grid = True):
    plt.plot (plot)
    plt.xlabel (x_label)
    plt.ylabel (y_label)
    plt.title (title)
    plt.grid (True) # This would show gridline at background 
    plt.show()

# Sample data (X and y)
x = np.array([1, 2, 3, 4, 5])   # Input features
y = np.array([3, 5, 7, 9, 11])  # Target values (perfect linear: y = 2x + 1)

x, y = LoadFile()
x, x_mean, x_std = normalize(x)
y, y_mean, y_std = normalize(y)

print("x np_array : ", x)
print("y np_array : ", y)
input("Loadfile continue 4")
# Initialize parameters
w = 0.0
b = 0.0

# Learning rate and number of iterations
learning_rate = 0.01
epochs = 1000
n = len(x)


# Training loop (Gradient Descent)
losses = []
prev_loss = float('inf')
for epoch in range(epochs):
    # Forward pass: compute predictions
    y_pred = w * x + b
    print("y_pred; ", y_pred)
    # input("enter continue...")


    # Compute the gradients
    dw = (1/n) * np.sum((y_pred - y) * x)
    db = (1/n) * np.sum(y_pred - y)

    # Optionally monitor gradient magnitude
    grad_magnitude = np.sqrt(dw**2 + db**2)
    if grad_magnitude < 1e-4:
        print(f"Gradients too small, stopping at epoch {epoch}")
        break
    
    # Update weights
    w -= learning_rate * dw
    b -= learning_rate * db
    
    # losses = []
    # Optionally print the loss every 100 steps
    if epoch % 100 == 0:
        loss = (1/n) * np.sum((y_pred - y)**2)
        # Check if loss improvement is small
        if abs(prev_loss - loss) < 0.000001:  # You can tune this threshold
            print(f"Converged at epoch {epoch}")
            break
        prev_loss = loss
        losses.append(loss)
        # print(f"Epoch {epoch}: Loss = {loss:.4f}, w = {w:.4f}, b = {b:.4f}")

    # draw_plt(loss, "loss", "weight", "Losses Over Weight", True)

    # plt.plot(losses)
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.title("Loss over Time")
    # plt.grid(True)
    # plt.show()

# plt.plot(losses)
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title("Loss over Time")
# plt.grid(True)
# plt.show()

# Rescale to original units (after training)
w_orig = y_std / x_std * w
b_orig = y_std * b + y_mean - w_orig * x_mean
print(f"Model in original scale: y = {w_orig:.2f}x + {b_orig:.2f}")

# print(f"\nFinal model: y = {w:.2f}x + {b:.2f}")
print("losses: ", losses)
draw_plt(losses, "Epoch", "Loss", "Loss Over Time", True)

# Final model parameters
