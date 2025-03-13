import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("darkgrid", {"grid.color": ".6", "grid.linestyle": ":"})

st.set_page_config(
    page_title="Stats Prediction",
)

st.markdown(
    """
      # Character Stats Prediction
      ## เดาสถานะตัวละครจากเลเวล
      โปรแกรมนี้จะทำการเดาสถานะตัวละครจากเลเวลของตัวละคร
      โดยใช้ข้อมูลจากเกม Genshin Impact ที่ได้จากการเก็บข้อมูลและทำการเรียนรู้จากข้อมูลนั้น
      ## Data Features
      - Lv: Level of the character
      - Base HP: Base HP of the character
      - Base ATK: Base ATK of the character
      - Base DEF: Base DEF of the character
      - CRIT DMG: CRIT DMG of the character
      - CRIT Rate: CRIT Rate of the character

      ref: https://github.com/Gusb3ll/keqing-stats-prediction
    """
)


def percent2float(x):
    try:
        x = float(x.strip("%")) / 100
        return x
    except:
        pass


train_data = pd.read_csv(
    "./data/stats/train/stats.csv",
    converters={"CRIT Rate": percent2float, "CRIT DMG": percent2float},
)
train_data = train_data.apply(pd.to_numeric, errors="coerce")
train_data = train_data.interpolate(method="linear")

st.dataframe(train_data)

# Plotting
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 8))


sns.scatterplot(x=train_data["Lv"], y=train_data["Base HP"], ax=ax1)
sns.scatterplot(x=train_data["Lv"], y=train_data["Base ATK"], ax=ax2)
sns.scatterplot(x=train_data["Lv"], y=train_data["Base DEF"], ax=ax3)
sns.scatterplot(x=train_data["Lv"], y=train_data["CRIT DMG"], ax=ax4)

st.pyplot(fig)

st.markdown(
    """
      ## Linear Regression
      ## 1. Import dataset & cleaning
    """
)

st.code(
    """
    def percent2float(x):
    try:
        x = float(x.strip("%")) / 100
        return x
    except:
        pass
        
    train_data = pd.read_csv(
      "./data/stats/train/stats.csv",
      converters={"CRIT Rate": percent2float, "CRIT DMG": percent2float},
    )
    train_data = train_data.apply(pd.to_numeric, errors="coerce")
    train_data = train_data.interpolate(method="linear")

    test_data = pd.read_csv(
      "./data/stats/test/stats.csv",
      converters={"CRIT Rate": percent2float, "CRIT DMG": percent2float},
    )
    test_data = test_data.apply(pd.to_numeric, errors="coerce")
    test_data = test_data.interpolate(method="linear")
    """,
    language="python",
)

st.markdown(
    """
      ## 2. Define model
    """
)


st.code(
    """
      in_d = (1,)

      def HP_model():
        model = tf.keras.Sequential(name='HP_Model',)
        model.add(tf.keras.layers.Dense(1, input_shape=in_d))
        model.compile(loss=tf.keras.losses.MeanAbsoluteError(), optimizer=tf.keras.optimizers.Adam(learning_rate=1))
        return model

      def ATK_model():
        model = tf.keras.Sequential(name='ATK_Model')
        model.add(tf.keras.layers.Dense(1, input_shape=in_d))
        model.compile(loss=tf.keras.losses.MeanAbsoluteError(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.05))
        return model

      def DEF_model():
        model = tf.keras.Sequential(name='DEF_Model')
        model.add(tf.keras.layers.Dense(1, input_shape=in_d))
        model.compile(loss=tf.keras.losses.MeanAbsoluteError(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.05))
        return model

      def CRIT_model():
        model = tf.keras.Sequential(name='CRIT_Model')
        model.add(tf.keras.layers.Dense(1, input_shape=in_d))
        model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.1))
        return model

      hp_m = HP_model()
      atk_m = ATK_model()
      def_m = DEF_model()
      crit_m = CRIT_model()
    """
)

st.markdown(
    """
      ## 3. Train model
    """
)

st.code(
    """
      index = train_df['Lv']

      hp_his = hp_m.fit(index, train_df['Base HP'], epochs=900, verbose=0)
      atk_his = atk_m.fit(index, train_df['Base ATK'], epochs=900, verbose=0)
      def_his = def_m.fit(index, train_df['Base DEF'], epochs=900, verbose=0)
      crit_his = crit_m.fit(index, train_df['CRIT DMG'], epochs=900, verbose=0)
    """
)

st.markdown("""## 4. Prediction""")

st.code(
    """
      result = {}

      result['hp_prediction'] = hp_m.predict(test_df['Lv'])
      result['atk_prediction'] = atk_m.predict(test_df['Lv'])
      result['def_prediction'] = def_m.predict(test_df['Lv'])
      result['crit_prediction'] = crit_m.predict(test_df['Lv']
    """
)

st.markdown(
    """
      ## 5. Save prediction result
    """
)

st.code(
    """
      HP = np.hstack(result['hp_prediction']).astype(int)
      ATK = np.hstack(result['atk_prediction']).astype(int)
      DEF = np.hstack(result['def_prediction']).astype(int)
      conv_crit = pd.DataFrame(result['crit_prediction']).applymap(lambda x: '{:.2%}'.format(x)).values
      CRIT = np.hstack(conv_crit).astype(str)

      data = {'Lv': test_df['Lv'], 'Base HP': HP, 'Base ATK': ATK, 'Base DEF': DEF, 'CRIT DMG': CRIT}

      result_df = pd.DataFrame(data, columns= ['Lv', 'Base HP', 'Base ATK', 'Base DEF', 'CRIT DMG'])

      result_df
    """
)

st.markdown(
    """
      ## Random Forest Regressor
      ## 1. Import dataset & cleaning
    """
)

st.code(
    """
    def percent2float(x):
    try:
        x = float(x.strip("%")) / 100
        return x
    except:
        pass
        
    train_data = pd.read_csv(
      "./data/stats/train/stats.csv",
      converters={"CRIT Rate": percent2float, "CRIT DMG": percent2float},
    )
    train_data = train_data.apply(pd.to_numeric, errors="coerce")
    train_data = train_data.interpolate(method="linear")

    test_data = pd.read_csv(
      "./data/stats/test/stats.csv",
      converters={"CRIT Rate": percent2float, "CRIT DMG": percent2float},
    )
    test_data = test_data.apply(pd.to_numeric, errors="coerce")
    test_data = test_data.interpolate(method="linear")
    """,
    language="python",
)

st.markdown(
    """
      ## 2. Define model
    """
)

st.code(
    """
      def HP_model():
          return RandomForestRegressor(
              n_estimators=9,
              max_depth=10,    
              random_state=42  
          )

      def ATK_model():
          return RandomForestRegressor(
              n_estimators=9,
              max_depth=10,
              random_state=42
          )

      def DEF_model():
          return RandomForestRegressor(
              n_estimators=9,
              max_depth=10,
              random_state=42
          )

      def CRIT_model():
          return RandomForestRegressor(
              n_estimators=9,
              max_depth=10,
              random_state=42
          )

      hp_m = HP_model()
      atk_m = ATK_model()
      def_m = DEF_model()
      crit_m = CRIT_model()
    """
)

st.markdown(
    """
      ## 3. Train model
    """
)

st.code(
    """
      X = train_df["Lv"].values.reshape(-1, 1)

      hp_his = hp_m.fit(X, train_df["Base HP"])
      atk_his = atk_m.fit(X, train_df["Base ATK"])
      def_his = def_m.fit(X, train_df["Base DEF"])
      crit_his = crit_m.fit(X, train_df["CRIT DMG"])
    """
)

st.markdown("""## 4. Prediction""")

st.code(
    """
      # Example prediction
      X_test = np.array([[0.5]])  # Single test input

      hp_pred = hp_m.predict(X_test)
      atk_pred = atk_m.predict(X_test)
      def_pred = def_m.predict(X_test)
      crit_pred = crit_m.predict(X_test)

      print("HP Prediction:", hp_pred)
      print("ATK Prediction:", atk_pred)
      print("DEF Prediction:", def_pred)
      print("CRIT Prediction:", crit_pred)
    """
)
