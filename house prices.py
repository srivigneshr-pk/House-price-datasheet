# Step 1: Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load Dataset
df = pd.read_csv(r"C:\Users\Reference\Desktop\House price prediction\usa.csv")  # Ensure the CSV is in the same folder

# Step 3: Select Features and Target
X = df[['bedrooms', 'bathrooms', 'sqft_living']]
y = df['price']


# Step 4: Split Data into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Train the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Predict and Evaluate
y_pred = model.predict(X_test)


# Step 7: User Input
print("\nEnter house details to predict the price:")
bedrooms = int(input("Enter number of bedrooms: "))
bathrooms = int(input("Enter number of bathrooms: "))
sqft = int(input("Enter square feet: "))

# Step 8: Predict Based on User Input
input_data = pd.DataFrame([[bedrooms, bathrooms, sqft]], columns=['bedrooms', 'bathrooms', 'sqft_living'])
predicted_price = model.predict(input_data)
print(f"\nPredicted House Price: ${predicted_price[0]:,.2f}")

# Step 9: Plot Actual vs Predicted Prices (for Test Set)
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.show()
