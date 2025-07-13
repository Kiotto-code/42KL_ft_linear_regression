Sure! Let's walk through a concrete example using **mileage vs. price of a car** to see how gradient descent works step-by-step — including how to calculate the gradient `dw = (1/n) * sum(X * error)` and `db = (1/n) * sum(error)`.

---

## 🚗 Problem Setup

You're trying to model:

> **Price of a used car** based on its **mileage** (in 1000s of miles)

Let’s assume a simple **linear regression model**:

$$
\hat{y} = wX + b
$$

Where:

* $X$: mileage (in thousands of miles)
* $y$: price (in \$1000s)
* $w$: weight (slope)
* $b$: bias (intercept)

---

## 🧪 Sample Data

| Mileage $X$ (1000s miles) | Price $y$ (in \$1000s) |
| ------------------------- | ---------------------- |
| 10                        | 20                     |
| 20                        | 18                     |
| 30                        | 15                     |

Let’s initialize:

* $w = 0$
* $b = 0$
* Learning rate = 0.01
* Number of data points $n = 3$

---

## 🔢 Step 1: Predict Prices

$$
\hat{y} = wX + b = 0 \cdot X + 0 = 0
$$

So predictions:

* $\hat{y}_1 = 0$
* $\hat{y}_2 = 0$
* $\hat{y}_3 = 0$

---

## 🧮 Step 2: Calculate Error

$$
\text{error} = \hat{y} - y
$$

$$
\text{error} = [0 - 20, 0 - 18, 0 - 15] = [-20, -18, -15]
$$

---

## 🔁 Step 3: Compute Gradients

### ✅ Compute `dw`:

$$
dw = \frac{1}{n} \sum (X \cdot \text{error})
$$

$$
X = [10, 20, 30], \quad \text{error} = [-20, -18, -15]
$$

$$
X \cdot \text{error} = [10 \cdot -20, 20 \cdot -18, 30 \cdot -15] = [-200, -360, -450]
$$

$$
dw = \frac{1}{3} (-200 - 360 - 450) = \frac{-1010}{3} \approx -336.67
$$

---

### ✅ Compute `db`:

$$
db = \frac{1}{n} \sum \text{error} = \frac{-20 -18 -15}{3} = \frac{-53}{3} \approx -17.67
$$

---

## 🛠️ Step 4: Update Parameters

Use:

$$
w = w - \text{learning rate} \cdot dw
$$

$$
b = b - \text{learning rate} \cdot db
$$

$$
w = 0 - 0.01 \cdot (-336.67) = 3.3667
$$

$$
b = 0 - 0.01 \cdot (-17.67) = 0.1767
$$

---

## ✅ Updated Model

$$
\hat{y} = 3.3667 \cdot X + 0.1767
$$

This model is now **starting to learn** a relationship between mileage and price.

---

Would you like to see the next iteration too? Or visualize this with a graph?
