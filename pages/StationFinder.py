import folium
import streamlit as st
import pandas as pd
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static

st.set_page_config(
    page_title="Train Station Finder",
)

st.markdown(
    """
      # Train Station Finder
      ## ตามหาสถานีรถไฟใกล้ฉัน
      โมเดลนี้จะทำการทำนายสายรถไฟที่ใกล้สุดจากตำแหน่งที่อยู่ของคุณ
      โดยใช้ข้อมูลจากการเรียนรู้ของตำแหน่งสถานีรถไฟในกรุงเทพมหานคร
      ## Data Features
      - name: Station name in Thai
      - nameEng: Station name in English
      - geoLat: Latitude of the station
      - geoLng: Longitude of the station
      - lineName: Line name of the station in Thai
      - lineNameEng: Line name in English
      - lineColorHex: Line color in HEX
      - lineServiceName: Line service name in Thai

      ref: https://www.kaggle.com/datasets/gusbell/thailand-public-train-data-bangkok-area
      
      ref2: https://github.com/Gusb3ll/thailand-public-train-data
    """
)

data = pd.read_json(
    "./data/station/data.json",
)
df = pd.DataFrame(data)

m = folium.Map(location=[13.736384, 100.636139], zoom_start=12)
marker_cluster = MarkerCluster().add_to(m)

for _, row in df.iterrows():
    folium.Marker(
        location=[row["geoLat"], row["geoLng"]],
        popup=f"Line: {row['lineServiceName']}",
        tooltip=row["lineServiceName"],
        icon=folium.Icon(color="blue", icon="train", prefix="fa"),
    ).add_to(marker_cluster)

folium_static(m)

st.markdown(
    """
      ## 1. Import dataset & cleaning
    """
)

st.code(
    """
      import pandas as pd
      from sklearn.preprocessing import StandardScaler, LabelEncoder
      from sklearn.model_selection import train_test_split
      import numpy as np
      
      data = pd.read_json("./data/station/data.json")
      df = pd.DataFrame(data)

      # Example MRT - ท่าพระ
      df['target_name'] = df['lineServiceName'] + " - " + df['name']      

      scaler = StandardScaler()
      label_encoder = LabelEncoder()
      
      features = df[["geoLat", "geoLng"]]
      features_scaled = scaler.fit_transform(features)
      
      target = df['combined_name']
      target_encoded = label_encoder.fit_transform(target)

      X_train, X_test, y_train, y_test = train_test_split(
          features_scaled, target_encoded, test_size=0.2, random_state=42
      )
    """
)

st.markdown(
    """
      ## 2. Define model
    """
)

st.code(
    """
      model = tf.keras.models.Sequential(
        [
          tf.keras.layers.Dense(64, activation="relu", input_shape=(2,)),
          tf.keras.layers.Dense(32, activation="relu"),
          tf.keras.layers.Dense(len(label_encoder.classes_), activation="softmax"),
        ]
      )

      model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
      )
      
    """
)

st.markdown(
    """
      ## 3. Train model
    """
)

st.code(
    """
      history = model.fit(X_train, y_train, epochs=1500, batch_size=16, validation_split=0.2)

      loss, accuracy = model.evaluate(X_test, y_test)
      print(f"Test Accuracy: {accuracy * 100:.2f}%")
    """
)

st.markdown("""## 4. Prediction""")

st.code(
    """
      def predict_line_service(lat, lng):
        coords = np.array([[lat, lng]])
        coords_scaled = scaler.transform(coords)
        prediction = model.predict(coords_scaled)
        predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
        return predicted_label[0]

      lat_input, lng_input = 13.7563, 100.5018  # Example coordinates
      predicted_service = predict_line_service(lat_input, lng_input)
      
      print(f"Predicted Line Service Name: {predicted_service}")
    """
)
