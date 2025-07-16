import json
import csv
import os
import numpy as np
import matplotlib.pyplot as plt

EPOCH = 1000
DEBUG = False

current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, "./data.csv")

def plot_loss_and_prediction(x_data, y_actual, y_trained_predicted, losses):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Loss over epochs
    axes[0].plot(range(len(losses)), losses, color='green', label='Loss Line')
    axes[0].set_title("Loss Over Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True)
    axes[0].legend()

    # Plot 2: Price vs Mileage
    axes[1].scatter(x_data, y_actual, color='blue', label='Actual Data')
    axes[1].plot(x_data, y_trained_predicted, color='red', label='Prediction Line')
    axes[1].set_title("Price over Mileage")
    axes[1].set_xlabel("Mileage")
    axes[1].set_ylabel("Price")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.show()



def draw_plt (plot, x_label, y_label, title, grid = True, x_data = None, y_data = None):
    # plt.scatter(scatters)
    if x_data is not None and y_data is not None and plot is not None :
        plt.scatter(x_data, y_data, color='blue', label='Actual Data')
        # plt.plot(plot, color='red', label='Prediction Line')
        plt.plot(x_data, plot, color='red', label='Prediction Line')
    # if plot is not None and title=="Loss Over Time":
    #     plt.plot (plot)
    plt.xlabel (x_label)
    plt.ylabel (y_label)
    plt.title (title)
    plt.grid (True) # This would show gridline at background
    plt.legend()

    plt.show()

# def draw_plt(plot, x_label, y_label, title, grid=True, x_data=None, y_data=None):
#     plt.figure()
    
#     # Plot scatter if given
#     if x_data is not None and y_data is not None:
#         plt.scatter(x_data, y_data, color='blue', label='Actual Data')
    
#     # Plot line if x and prediction values are given
#     if x_data is not None and plot is not None and title != "Loss Over Time":
#         plt.plot(x_data, plot, color='red', label='Prediction Line')

#     # Plot loss over epochs
#     if plot is not None and title == "Loss Over Time":
#         plt.plot(range(len(plot)), plot, color='green', label='Loss')

#     plt.xlabel(x_label)
#     plt.ylabel(y_label)
#     plt.title(title)
#     if grid:
#         plt.grid(True)
#     plt.legend()
#     plt.show()


def normalize(arr):
    mean = np.mean(arr)
    std = np.std(arr)
    return (arr - mean) / std, mean, std

def ft_debug(msg):
    if DEBUG:
        input (msg)

def correlation_analysis(y_trained_prediction, actual_y, y_mean):
    return 1 - np.sum((y_trained_prediction - actual_y)**2) / np.sum((actual_y - y_mean)**2)

class TrainedModel:
# estimate Price(mileage) = θ0 + (θ1 ∗ mileage) / Linear Regression for prediction
    def __init__ (self, x=None, y=None, w_denormal=None, b_denormal=None):
        if x is None or y is None or w_denormal is None or b_denormal is None:
            self.load_model_from_json("TrainedModel.json")
        else:
            self.x_array = x
            self.y_array = y
            self.weight = w_denormal
            self.bias = b_denormal

    def MileageCalc (self):
        mileage = float(input("Enter the mileage (in km): "))
        print("You entered:", mileage)

        Price = mileage * self.weight + self.bias
        print(f"Predicted Price: {Price:.2f}")

    def load_model_from_json(self, filename):
        with open(filename, 'r') as f:
            model = json.load(f)

        # Convert lists back to NumPy arrays
        self.x_array = np.array(model['x'])
        self.y_array = np.array(model['y'])

        # Get model parameters
        self.weight = model['weight']
        self.bias = model['bias']

        # return x, y, weight, bias

    # def save_model_to_json(self, filename, x, y, w_denormal, b_denormal):
    #     model = {
    #         "x": x.tolist(),   # Convert np.ndarray to list
    #         "y": y.tolist(),
    #         "weight": float(w_denormal),  # Ensure float, not np.float64
    #         "bias": float(b_denormal)
    #     }
    #     with open(filename, 'w') as f:
    #         json.dump(model, f, indent=4)

    def LinearPrediction(self):
        return self.weight * self.x_array + self.bias

    def ModelPredict(self):
        y_trained_pred = self.LinearPrediction()
        draw_plt(y_trained_pred, "Epoch", "Price", "Price Over Mileage", True, self.x_array, self.y_array)
        # draw_plt(self.x_array, self.y_array, y_trained_pred)


class LinearRegressModel:
    def __init__(self):
        self.x = [] # The input rate of Mileage (The Gradient of SLR slope) -> Mileage
        self.y = [] # The final calculation of the Price via SLR calculation -> Price
        # self.weight = []
        # self.intercept = []
        self._modelData = []
        self.LoadFile()

    def save_model_to_json(self, filename, x, y, w_denormal, b_denormal):
        model = {
            "x": x.tolist(),   # Convert np.ndarray to list
            "y": y.tolist(),
            "weight": float(w_denormal),  # Ensure float, not np.float64
            "bias": float(b_denormal)
        }
        with open(filename, 'w') as f:
            json.dump(model, f, indent=4)

    def LoadFile(self):
        try:
                # print(list(model_data))
            x_vals = []
            y_vals = []
            with open(csv_path, 'r') as file:
                model_data = csv.DictReader(file)
                for r in model_data:
                    x_vals.append(float(r['km']))
                    y_vals.append(float(r['price']))
            self.x = np.array(x_vals)
            self.y = np.array(y_vals)
            # print (self._modelData)
            print("LoadtFile x : ", self.x)
            print("LoadFile y : ", self.y)
            ft_debug("Loadfile continue 3")
            return np.array(self.x), np.array(self.y)
        except Exception as e:
            # An unopenable file
            print(f"Error loading file: {e}")



    # def SLR_calculation(self):
    def GradientDescent(self):
        x, y = self.x, self.y
        # x, y = np.array(self.x), np.array(self.y)
        if x is None or y is None:
            print("Training data is Empty!")
            return False
        normaliezd_x, x_mean, x_std = normalize(x)
        normaliezd_y, y_mean, y_std = normalize(y)
        print("x", normaliezd_x)
        print("y", normaliezd_y)
        w = 0.0
        b = 0.0

        learning_rate = 0.01
        epochs = 1000 # This is when algo has complate pass the data set once
        ft_debug(f"normalized x: {normaliezd_x}")
        n = len(normaliezd_x) # This is the number data case for calculate the mean in MSE
        print("n: ", n)
        ft_debug("continue")
        losses = [] # append the losses to calculate losses overtime
        prev_loss = float('inf')
        for epoch in range(epochs):
            y_untrained_pred = w * normaliezd_x + b
            print("y_untrained_pred: ", y_untrained_pred)

            loss = 1/n * np.sum((y_untrained_pred  - normaliezd_y) ** 2) # MSE cost function for finding convergence
            if abs(prev_loss - loss) < 1e-6:
                print(f"Converged at epoch {epoch}\n" +
                f"pred_loss, loss: {prev_loss}, {loss}\n" +
                f"loss_diff: {prev_loss - loss:.20f}"
                )
                break
            prev_loss = loss
            losses.append(loss)

            dw = (1/n) * np.sum((y_untrained_pred - normaliezd_y) * normaliezd_x)
            db = (1/n) * np.sum(y_untrained_pred - normaliezd_y)

            # check the error gradient magnitude using Eucledian Norm
            grad_magnitude = np.sqrt(dw**2 + db**2)
            if grad_magnitude < 1e-4:
                print("The Error Gradient is low enough")
                break

            w -= learning_rate * dw
            b -= learning_rate * db

            # Optionally print the loss every 100 steps
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.4f}, w = {w:.4f}, b = {b:.4f}")
        
        print("losses: ", losses)
        print(f"Final model normaled: y = {w:.2f}x + {b:2f}\n")

        # Rescale to original units (after training)
        w_denormal = y_std / x_std * w
        b_denormal = y_std * b + y_mean - w_denormal * x_mean
        print(f"Final model after denormal: y = {w_denormal:.2f}x + {b_denormal:2f}\n")

        y_trained_pred = w_denormal * self.x + b_denormal
        plot_loss_and_prediction(self.x, self.y, y_trained_pred, losses)

        # print("y_array prediction after trained; ", y_trained_pred)
        # ft_debug("Continue...")
        # print("x_array prediction after trained; ", x_array)
        # ft_debug("Continue...")
        # print(f"Model in original scale: y = {w_denormal:.2f}x + {b_denormal:.2f}")

        # self.trained_model = TrainedModel(self.x, self.y, w_denormal, b_denormal)
        print("correlation_analysis: ", correlation_analysis(y_trained_pred, np.array(y), y_mean))
        self.save_model_to_json("TrainedModel.json", self.x, self.y, w_denormal, b_denormal)

if __name__ == "__main__":
    try:
        model = LinearRegressModel()
        model.GradientDescent()

        PredictionModel = TrainedModel()
        PredictionModel.ModelPredict()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


"""
# With lists (slower)
a = [1, 2, 3]
b = [4, 5, 6]
c = [x + y for x, y in zip(a, b)]

# With NumPy (faster)
import numpy as np
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = a + b  # cleaner and faster
"""