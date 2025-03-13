import joblib
import streamlit as st
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})

st.set_page_config(
    page_title="Stats Prediction Model",
)

st.markdown(
    """
      # Character Stats Prediction Model
    """
)

choices = st.selectbox(
    "Select model",
    ("Select your model", "Linear Regression", "Random Forest Regressor"),
)


def percent2float(x):
    try:
        x = float(x.strip("%")) / 100
        return x
    except:
        pass


data = pd.read_csv(
    "./data/stats/test/stats.csv",
    converters={"CRIT Rate": percent2float, "CRIT DMG": percent2float},
)


if choices == "Linear Regression":
    st.markdown(
        """
        ## Linear Regression
        """
    )

    hp_m = tf.keras.models.load_model("./data/models/stats/linear/HP.keras")
    atk_m = tf.keras.models.load_model("./data/models/stats/linear/ATK.keras")
    def_m = tf.keras.models.load_model("./data/models/stats/linear/DEF.keras")
    crit_m = tf.keras.models.load_model("./data/models/stats/linear/CRIT.keras")

    result = {}

    result["hp_prediction"] = hp_m.predict(data["Lv"])
    result["atk_prediction"] = atk_m.predict(data["Lv"])
    result["def_prediction"] = def_m.predict(data["Lv"])
    result["crit_prediction"] = crit_m.predict(data["Lv"])

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 8))

    sns.scatterplot(x=data["Lv"], y=data["Base HP"], ax=ax1)
    ax1.plot(data["Lv"], result["hp_prediction"], color="g")
    sns.scatterplot(x=data["Lv"], y=data["Base ATK"], ax=ax2)
    ax2.plot(data["Lv"], result["atk_prediction"], color="r")

    sns.scatterplot(x=data["Lv"], y=data["Base DEF"], ax=ax3)
    ax3.plot(data["Lv"], result["def_prediction"], color="b")

    sns.scatterplot(x=data["Lv"], y=data["CRIT DMG"], ax=ax4)
    ax4.plot(data["Lv"], result["crit_prediction"], color="b")

    st.pyplot(fig)

    data = data.apply(pd.to_numeric, errors="coerce")
    data = data.interpolate(method="linear")

    HP = np.hstack(result["hp_prediction"]).astype(int)
    ATK = np.hstack(result["atk_prediction"]).astype(int)
    DEF = np.hstack(result["def_prediction"]).astype(int)
    conv_crit = (
        pd.DataFrame(result["crit_prediction"])
        .applymap(lambda x: "{:.2%}".format(x))
        .values
    )
    CRIT = np.hstack(conv_crit).astype(str)
    conv_crit_rate = (
        pd.DataFrame(data["CRIT Rate"]).applymap(lambda x: "{:.2%}".format(x)).values
    )
    CRIT_RATE = np.hstack(conv_crit_rate).astype(str)

    data = {
        "Lv": data["Lv"],
        "Base HP": HP,
        "Base ATK": ATK,
        "Base DEF": DEF,
        "CRIT DMG": CRIT,
        "CRIT RATE": CRIT_RATE,
    }

    result = pd.DataFrame(
        data, columns=["Lv", "Base HP", "Base ATK", "Base DEF", "CRIT DMG", "CRIT RATE"]
    )

    result

elif choices == "Random Forest Regressor":
    st.markdown(
        """
        ## Random Forest Regressor
        """
    )

    level = st.slider("Select level", 1, 100, 1)

    hp_m = joblib.load("./data/models/stats/polynomial/hp_model.pkl")
    atk_m = joblib.load("./data/models/stats/polynomial/atk_model.pkl")
    def_m = joblib.load("./data/models/stats/polynomial/def_model.pkl")
    crit_m = joblib.load("./data/models/stats/polynomial/crit_model.pkl")

    val = np.array([[level]])

    hp_pred = hp_m.predict(val)
    atk_pred = atk_m.predict(val)
    def_pred = def_m.predict(val)
    crit_pred = crit_m.predict(val)

    st.text(f"HP Prediction: {hp_pred}")
    st.text(f"ATK Prediction: {atk_pred}")
    st.text(f"DEF Prediction: {def_pred}")
    st.text(f"CRIT Prediction: {crit_pred}")
