# estimateP rice(mileage) = θ0 + (θ1 ∗ mileage)
from GradientDescent import *


def MileagePredict():
    PredictModel = TrainedModel()
    PredictModel.MileageCalc()


if __name__ == "__main__":
    while(True):
        try:
            MileagePredict()
        except Exception as e:
            print(f"An unexpected error occurred: {e}")