


from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from utils.data_io import load_clean_data

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE 
from sklearn.preprocessing import StandardScaler 
import joblib 
import pandas as pd 
import preprocessing.data_preprocessing as process 


def compute_VIF(X_encoded):
    X_encoded = X_encoded.astype(float)
    X_with_const = add_constant(X_encoded)
    vif_data = pd.DataFrame()
    vif_data["feature"] = X_encoded.columns
    vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i+1) for i in range(X_encoded.shape[1])]
    return vif_data


def preprocess_data_for_LR():
    """ """
    data = load_clean_data()
    data = process.drop_insignificant_features(data)
    X, y = process.split_features_target(data)
    X = process.encode_categoricals(X)
    X_train, X_test, y_train, y_test = process.train_test_stratified_split(X, y)
    
    print("🔍 Computing VIFs...")
    vif_df = compute_VIF(X_train)
    print(vif_df.sort_values("VIF", ascending=False))
    

if __name__ == '__main__':
    preprocess_data_for_LR()