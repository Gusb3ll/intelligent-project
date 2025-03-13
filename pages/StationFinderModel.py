import folium
import streamlit as st
import pandas as pd
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

st.set_page_config(
    page_title="Train Station Finder Model",
)

st.markdown(
    """
      # Train Station Finder Model
    """
)

lat = st.text_input("Latitude", value=13.736384)
lon = st.text_input("Longitude", value=100.636139)

data = pd.read_json("./data/station/data.json")
df = pd.DataFrame(data)

features = df[["geoLat", "geoLng"]]
target = df["lineServiceName"]

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

label_encoder = LabelEncoder()
target_encoded = label_encoder.fit_transform(target)

model = tf.keras.models.load_model("./data/models/station/model.keras")


def predict_line(lat, lng):
    coords = np.array([[lat, lng]])
    coords_scaled = scaler.transform(coords)
    prediction = model.predict(coords_scaled)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])

    return predicted_label[0]


result = predict_line(lat, lon)

st.write(f"Predicted line service: {result}")

m = folium.Map(location=[lat, lon], zoom_start=12)


for _, row in df.iterrows():
    tooltip_1 = row["lineServiceName"]
    tooltip_2 = row["name"]

    folium.Marker(
        location=[row["geoLat"], row["geoLng"]],
        popup=f"Line: {row['lineServiceName']}",
        tooltip=f"{tooltip_1} - {tooltip_2}",
        icon=folium.Icon(color="blue", icon="train", prefix="fa"),
    ).add_to(m)


folium.Marker(
    location=[float(lat), float(lon)],
    popup="Your location",
    icon=folium.Icon(color="red", icon="home", prefix="fa"),
).add_to(m)

folium_static(m)
