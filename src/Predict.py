# estimateP rice(mileage) = θ0 + (θ1 ∗ mileage)

class SimpleLinear:
    def __init__(self):
        self.θ0 = 0.0 # Initial Price of the Model (When the Input var is 0, the y-intercept)
        self.θ1 = 0.0 # The Gradient of SLR (price changes for each unit x)
        self.mileage = 0.0 # The input rate of Mileage (The Gradient of SLR slope)
        self.EstimatedPrice = 0.0 # The final calculation of the Price via SLR calculation

    def SLR_calculation(self):
        
