import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Correct dataset
hours = np.array([1,2,3,4,5,6,7,8,9,10], dtype=float)
scores = np.array([15,25,35,45,55,65,75,85,95,105], dtype=float)  # Linear: 10*x + 5

# Build model
model = Sequential([Dense(1, input_shape=[1])])

# Compile model
model.compile(optimizer='SGD', loss='mean_squared_error')

# Train model
model.fit(hours, scores, epochs=500, verbose=0)

# Save model in the same folder
model.save("student_score.h5")
print("Model saved as student_score.h5")

# Test predictions
print("Prediction for 5 hours:", model.predict(np.array([[5]], dtype=float))[0][0])
print("Prediction for 10 hours:", model.predict(np.array([[10]], dtype=float))[0][0])

