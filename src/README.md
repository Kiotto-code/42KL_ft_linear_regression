The First Program required to do the Prediction (Simple Linear Regression)
    - estimateP rice(mileage) = θ0 + (θ1 ∗ mileage) (Similar to y = mx + c)
    - Simpe cause there is only one input variable.

Cost Function:
Great question! You're asking where the formula for the **gradient of the loss with respect to the weight**,

$$
dw = \frac{1}{n} \sum (X \cdot \text{error})
$$

comes from.

Let’s walk through **how this is derived** step by step from first principles — using **calculus** and the **Mean Squared Error (MSE)** loss function.

---

## 🧠 Setup: The Model

We are using a simple **linear regression model**:

$$
\hat{y}_i = wX_i + b
$$

Where:

* $X_i$ is the input (e.g., mileage)
* $\hat{y}_i$ is the predicted value
* $y_i$ is the true (actual) value

---

## 🎯 Goal: Minimize the **Loss Function**

We use the **Mean Squared Error (MSE)**:

$$
\text{Loss} = \frac{1}{n} \sum_{i=1}^n (\hat{y}_i - y_i)^2
$$

Plug in the prediction formula:

$$
\text{Loss} = \frac{1}{n} \sum_{i=1}^n (wX_i + b - y_i)^2
$$

We want to compute the **partial derivative of the loss with respect to $w$**:

$$
\frac{\partial \text{Loss}}{\partial w}
$$

---

## 🧮 Step-by-Step Derivation

Let’s define the error:

$$
e_i = \hat{y}_i - y_i = (wX_i + b - y_i)
$$

Then the loss becomes:

$$
\text{Loss} = \frac{1}{n} \sum e_i^2
$$

Now apply the chain rule:

$$
\frac{\partial \text{Loss}}{\partial w} = \frac{1}{n} \sum_{i=1}^n \frac{\partial}{\partial w} (e_i^2)
$$

$$
= \frac{1}{n} \sum_{i=1}^n 2e_i \cdot \frac{\partial e_i}{\partial w}
$$

Recall:

$$
e_i = wX_i + b - y_i
\quad \Rightarrow \quad
\frac{\partial e_i}{\partial w} = X_i
$$

So:

$$
\frac{\partial \text{Loss}}{\partial w} = \frac{1}{n} \sum_{i=1}^n 2(wX_i + b - y_i) \cdot X_i
$$

Factor out the 2:

$$
= \frac{2}{n} \sum X_i \cdot (wX_i + b - y_i)
$$

✅ This is the exact derivative. But in practice, we often **drop the 2** (absorbed into learning rate), leading to:

$$
dw = \frac{1}{n} \sum X_i \cdot (\hat{y}_i - y_i)
$$


---

## 🧠 Final Formula:

$$
\boxed{dw = \frac{1}{n} \sum X_i \cdot (wX_i + b - y_i)}
$$

This gives us the **gradient** of the loss with respect to the weight — and tells us how to adjust `w` to reduce the loss.

---

## Check Correlation of Determination:
$$
R2=1−∑(y−yˉ​)2∑(ypred​−y)2​
$$

---

[Linear Regression with Gradient Descent Derivation](https://medium.com/analytics-vidhya/linear-regression-with-gradient-descent-derivation-c10685ddf0f4)