import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Generate dataset
X, y = make_regression(n_samples=100, n_features=1, n_informative=1, n_targets=1, noise=20, random_state=13)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

# Custom GD Regressor
class GDRegressor:

    def __init__(self, learning_rate, epochs):
        self.m = 100  # Initial guess for m (slope)
        self.b = -120 # Initial guess for b (intercept)
        self.lr = learning_rate
        self.epochs = epochs

    def fit(self, X, y):
        for i in range(self.epochs):
            loss_slope_b = -2 * np.sum(y - self.m * X.ravel() - self.b)
            loss_slope_m = -2 * np.sum((y - self.m * X.ravel() - self.b) * X.ravel())
            self.b = self.b - (self.lr * loss_slope_b)
            self.m = self.m - (self.lr * loss_slope_m)
        print(f"Final m: {self.m}, Final b: {self.b}")

    def predict(self, X):
        return self.m * X + self.b

# Initialize and train the model
gd = GDRegressor(0.001, 50)
gd.fit(X_train, y_train)

# Predict and evaluate
y_pred = gd.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"R-squared: {r2}")

# Create 3D plot for visualization
m_values = np.linspace(gd.m - 20, gd.m + 20, 100)
b_values = np.linspace(gd.b - 20, gd.b + 20, 100)
M, B = np.meshgrid(m_values, b_values)

def calculate_loss(m, b, X, y):
    return np.sum((y - m * X.ravel() - b)**2)

loss_values = np.zeros_like(M)
for i in range(len(m_values)):
    for j in range(len(b_values)):
        loss_values[i, j] = calculate_loss(M[i, j], B[i, j], X_train, y_train)

# 3D Plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(M, B, loss_values, cmap='viridis', alpha=0.7)

ax.scatter(gd.m, gd.b, calculate_loss(gd.m, gd.b, X_train, y_train), color='red', marker='o', s=100, label='Final m, b')

ax.set_xlabel('m (Slope)')
ax.set_ylabel('b (Intercept)')
ax.set_zlabel('Loss')
ax.set_title('Gradient Descent Visualization')
ax.legend()
plt.show()
