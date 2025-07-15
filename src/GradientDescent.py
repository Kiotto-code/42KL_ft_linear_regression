import json
import csv
import os
import numpy as np
import matplotlib.pyplot as plt

EPOCH = 1000


current_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_dir, "./car_data.csv")

def draw_plt (plot, x_label, y_label, title, grid = True, x_data = None, y_data = None):
    # plt.scatter(scatters)
    if x_data is not None and y_data is not None and plot is not None :
        plt.scatter(x_data, y_data, color='blue', label='Actual Data')
        # plt.plot(plot, color='red', label='Prediction Line')
        plt.plot(x_data, plot, color='red', label='Prediction Line')
    if plot is not None and title=="Loss Over Time":
        plt.plot (plot)
    plt.xlabel (x_label)
    plt.ylabel (y_label)
    plt.title (title)
    plt.grid (True) # This would show gridline at background 
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

class TrainedModel:
# estimate Price(mileage) = θ0 + (θ1 ∗ mileage) / Linear Regression for prediction
    def __init__ (self, x, y, m, c):
        self.x = x # Initial Price of the Model (When the Input var is 0, the y-intercept)
        self.y = y # The Gradient of SLR (price changes for each unit x)
        self.m = m
        self.c = c
    # def load_model(self):

class LinearRegressModel:
    def __init__(self):
        self.x = [] # The input rate of Mileage (The Gradient of SLR slope) -> Mileage
        self.y = [] # The final calculation of the Price via SLR calculation -> Price
        # self.weight = []
        # self.intercept = []
        self._modelData = []
        self.LoadFile()

    def LoadFile(self):
        try:
            with open(csv_path, 'r') as file:
                model_data = csv.DictReader(file)
                # print(list(model_data))
                for r in model_data:
                    self._modelData.append(r)
                    self.x.append(float(r['mileage']))  # Mileage is the km traveled by vehicles
                    self.y.append(float(r['price'])) # Price is the Price of car after traveled Mileage
            # print (self._modelData)
            print("LoadtFile x : ", self.x)
            print("LoadFile y : ", self.y)
            input("Loadfile continue 3")
            return np.array(self.x), np.array(self.y)
        except Exception as e:
            # An unopenable file
            print(f"Error loading file: {e}")

    @staticmethod
    def linear_prediction(weight, givenMileage, bias):
        # This is to predict the Predicted y(Price) and x(Mileage) -> "y=mx+c" -> Program 1
        y = weight * givenMileage + bias
        return y

    # def SLR_calculation(self):
    def GradientDescent(self):
        x, y = np.array(self.x), np.array(self.y)
        x, x_mean, x_std = normalize(x)
        y, y_mean, y_std = normalize(y)
        print("x", x)
        print("y", y)
        if x is None or y is None:
            # When given empty sets of data
            print("Training data is Empty!")
            return False
        w = 0.0
        b = 0.0

        learning_rate = 0.01
        epochs = 1000 # This is when algo has complate pass the data set once
        n = len(x) # This is the number data case for calculate the mean in MSE
        print("n: ", n)
        input("continue")
        losses = [] # append the losses to calculate losses overtime
        prev_loss = float('inf')
        for epoch in range(epochs):

            # y_pred = self.linear_prediction(w, x, b) # find the y-prediction based on the linear line
            y_pred = w * x + b
            print("y_pred: ", y_pred)


            loss = 1/n * np.sum((y_pred  - y) ** 2) # MSE cost function for finding convergence
            if abs(prev_loss - loss) < 1e-6:
                print(f"Converged at epoch {epoch}\n" +
                f"pred_loss, loss: {prev_loss}, {loss}\n" +
                f"loss_diff: {prev_loss - loss:.20f}"
                )
                break
            prev_loss = loss
            losses.append(loss)

            dw = (1/n) * np.sum((y_pred - y) * x)
            db = (1/n) * np.sum(y_pred - y)

            # check the error gradient magnitude using Eucledian Norm
            grad_magnitude = np.sqrt(dw**2 + db**2)
            if grad_magnitude < 1e-4:
                print("The Error Gradient is low enough")
                break

            w -= learning_rate * dw
            b -= learning_rate * db

            # Optionally print the loss every 100 steps
            if epoch % 1000 == 0:
                # loss = (1/n) * np.sum((y_pred - y)**2)
                # print("loss: ", loss)
                # if abs(prev_loss - loss) < 1e-6:  # You can tune this threshold
                #     print(f"Converged at epoch {epoch}")
                #     break
                # prev_loss = loss
                # losses.append(loss)
                # print("losses: ", losses)
                # input("continue")
                print(f"Epoch {epoch}: Loss = {loss:.4f}, w = {w:.4f}, b = {b:.4f}")

        
        draw_plt(losses, "Epoch", "Loss", "Loss Over Time", True)
        print("losses: ", losses)
        print(f"Final model: y = {w:.2f}x + {b:.2f}\n")

        # Rescale to original units (after training)
        w_orig = y_std / x_std * w
        b_orig = y_std * b + y_mean - w_orig * x_mean

        # Prices = []
        # fot case in range()

        x_array = np.array(self.x)
        y_pred_orig = w_orig * x_array + b_orig
        draw_plt(y_pred_orig, "Mileage", "Price", "Price over Mileage", True, x_array, self.y) # This would show gridline at background 
        print("y_array prediction after trained; ", y_pred_orig)
        input()
        print("x_array prediction after trained; ", x_array)
        input()
        print(f"Model in original scale: y = {w_orig:.2f}x + {b_orig:.2f}")

        # self.DescentedLinear = TrainedModel(x, y, m, c)
        

if __name__ == "__main__":
    model = LinearRegressModel()
    model.GradientDescent()
    
    