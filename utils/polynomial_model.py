import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


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


def HP_model():
    return RandomForestRegressor(n_estimators=9, max_depth=10, random_state=42)


def ATK_model():
    return RandomForestRegressor(n_estimators=9, max_depth=10, random_state=42)


def DEF_model():
    return RandomForestRegressor(n_estimators=9, max_depth=10, random_state=42)


def CRIT_model():
    return RandomForestRegressor(n_estimators=9, max_depth=10, random_state=42)


hp_m = HP_model()
atk_m = ATK_model()
def_m = DEF_model()
crit_m = CRIT_model()

X = train_df["Lv"].values.reshape(-1, 1)

hp_his = hp_m.fit(X, train_df["Base HP"])
atk_his = atk_m.fit(X, train_df["Base ATK"])
def_his = def_m.fit(X, train_df["Base DEF"])
crit_his = crit_m.fit(X, train_df["CRIT DMG"])

# Save model

joblib.dump(hp_his, "hp_model.pkl")
joblib.dump(atk_his, "atk_model.pkl")
joblib.dump(def_his, "def_model.pkl")
joblib.dump(crit_his, "crit_model.pkl")
