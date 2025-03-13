import pandas as pd
import tensorflow as tf


def percent2float(x):
    try:
        x = float(x.strip("%")) / 100
        return x
    except:
        pass


train_df = pd.read_csv(
    "./data/stats/train/stats.csv",
    converters={"CRIT Rate": percent2float, "CRIT DMG": percent2float},
)
train_df = train_df.apply(pd.to_numeric, errors="coerce")
train_df = train_df.interpolate(method="linear")

test_df = pd.read_csv(
    "./data/stats/test/stats.csv",
    converters={"CRIT Rate": percent2float, "CRIT DMG": percent2float},
)

in_d = (1,)


def HP_model():
    model = tf.keras.Sequential(
        name="HP_Model",
    )
    model.add(tf.keras.layers.Dense(1, input_shape=in_d))
    model.compile(
        loss=tf.keras.losses.MeanAbsoluteError(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=1.0),
    )
    return model


def ATK_model():
    model = tf.keras.Sequential(name="ATK_Model")
    model.add(tf.keras.layers.Dense(1, input_shape=in_d))
    model.compile(
        loss=tf.keras.losses.MeanAbsoluteError(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
    )
    return model


def DEF_model():
    model = tf.keras.Sequential(name="DEF_Model")
    model.add(tf.keras.layers.Dense(1, input_shape=in_d))
    model.compile(
        loss=tf.keras.losses.MeanAbsoluteError(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
    )
    return model


def CRIT_model():
    model = tf.keras.Sequential(name="CRIT_Model")
    model.add(tf.keras.layers.Dense(1, input_shape=in_d))
    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    )
    return model


hp_m = HP_model()
atk_m = ATK_model()
def_m = DEF_model()
crit_m = CRIT_model()

X = train_df["Lv"]

hp_his = hp_m.fit(X, train_df["Base HP"], epochs=900, verbose=1)
atk_his = atk_m.fit(X, train_df["Base ATK"], epochs=900, verbose=1)
def_his = def_m.fit(X, train_df["Base DEF"], epochs=900, verbose=1)
crit_his = crit_m.fit(X, train_df["CRIT DMG"], epochs=900, verbose=1)

# Save model

hp_m.save("./models/HP.keras")
atk_m.save("./models/ATK.keras")
def_m.save("./models/DEF.keras")
crit_m.save("./models/CRIT.keras")
