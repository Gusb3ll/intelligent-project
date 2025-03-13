import pandas as pd
import requests
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf


# Load data from URL
url = "https://raw.githubusercontent.com/Gusb3ll/thailand-public-train-data/refs/heads/main/dist/data.json"
response = requests.get(url)
data = response.json()

# Convert JSON to DataFrame
df = pd.DataFrame(data)

# import folium


# m = folium.Map(location=[13.736384, 100.636139], zoom_start=12)

# for _, row in df.iterrows():
#     folium.Marker(
#         location=[row["geoLat"], row["geoLng"]],
#         popup=f"Line: {row['lineServiceName']}",
#         icon=folium.Icon(color="blue", icon="train", prefix="fa"),
#     ).add_to(m)

# m.save("train_map.html")

features = df[["geoLat", "geoLng"]]
target = df["lineServiceName"]

label_encoder = LabelEncoder()
target_encoded = label_encoder.fit_transform(target)

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    features_scaled, target_encoded, test_size=0.2, random_state=42
)

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Dense(64, activation="relu", input_shape=(2,)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(len(label_encoder.classes_), activation="softmax"),
    ]
)

# Compile the model
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

history = model.fit(X_train, y_train, epochs=1000, batch_size=16, validation_split=0.2)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

model.save("./models/station.keras")

# def predict_line_service(lat, lng):
#     coords = np.array([[lat, lng]])
#     coords_scaled = scaler.transform(coords)
#     prediction = model.predict(coords_scaled)
#     predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
#     return predicted_label[0]


# # Example prediction
# lat_input, lng_input = 13.7563, 100.5018  # Example coordinates
# predicted_service = predict_line_service(lat_input, lng_input)
# print(f"Predicted Line Service Name: {predicted_service}")
