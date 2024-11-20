import matplotlib.pyplot as plt

# Store metrics for plotting
degrees = range(1, 5)
r2_scores = []
mae_scores = []
mse_scores = []

# Loop through polynomial degrees
for degree in degrees:
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)
    y_pred_poly = poly_model.predict(X_test_poly)
    
    mae = mean_absolute_error(y_test, y_pred_poly)
    mse = mean_squared_error(y_test, y_pred_poly)
    r2 = r2_score(y_test, y_pred_poly)
    
    # Append scores for plotting
    r2_scores.append(r2)
    mae_scores.append(mae)
    mse_scores.append(mse)
    
    # Print results for the degree
    print(f"Degree {degree}:")
    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"R^2: {r2}")
    print("-" * 30)
    
    # Plot actual vs predicted
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred_poly, alpha=0.6, color='blue', label='Predicted vs Actual')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Perfect Prediction')
    plt.title(f"Polynomial Regression (degree={degree}): Actual vs Predicted")
    plt.xlabel("Actual Sales")
    plt.ylabel("Predicted Sales")
    plt.legend()
    plt.grid(True)
    plt.show()

# Bar graph for R^2, MAE, MSE
plt.figure(figsize=(12, 6))

# R^2 scores
plt.subplot(1, 3, 1)
plt.bar(degrees, r2_scores, color='green', alpha=0.7)
plt.title('R^2 Scores by Degree')
plt.xlabel('Polynomial Degree')
plt.ylabel('R^2 Score')
plt.xticks(degrees)

# MAE
plt.subplot(1, 3, 2)
plt.bar(degrees, mae_scores, color='orange', alpha=0.7)
plt.title('MAE by Degree')
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Absolute Error')
plt.xticks(degrees)

# MSE
plt.subplot(1, 3, 3)
plt.bar(degrees, mse_scores, color='blue', alpha=0.7)
plt.title('MSE by Degree')
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error')
plt.xticks(degrees)

plt.tight_layout()
plt.show()


