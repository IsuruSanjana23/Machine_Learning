import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#medical_df = pd.read_csv('medical.csv')

def After_Data_Split(Dataframe):
    enc = preprocessing.OneHotEncoder()
    enc.fit(Dataframe[['region']])
    one_hot = enc.transform(Dataframe[['region']])
    one_hot = one_hot.toarray()
    Dataframe[['northeast', 'northwest', 'southeast', 'southwest']] = one_hot

    smoker_codes = {'no' : 0 , 'yes' : 1 }
    sex_codes = {'male' : 1 , 'female' : 0 }
    Dataframe['smoker_code'] = Dataframe['smoker'].map(smoker_codes)
    Dataframe['sex_code'] = Dataframe['sex'].map(sex_codes)

    numeric_cols = ['age', 'bmi', 'children']
    cat_cols = ['smoker_code', 'sex_code', 'northeast', 'northwest', 'southeast', 'southwest']

    numeric_data = Dataframe[numeric_cols].values
    categorical_data = Dataframe[cat_cols].values
    inputs = np.concatenate((numeric_data, categorical_data), axis=1)
    targets = Dataframe['charges']

    # 1️⃣ Split first (optional — or scale before splitting)
    inputs_train, inputs_test, targets_train, targets_test = train_test_split(
        inputs, targets, test_size=0.1, random_state=42
    )

    scaler = StandardScaler()

    # 2️⃣ Fit the scaler on the training data
    input_train_scaled = scaler.fit_transform(inputs_train)

    # 3️⃣ Transform the test data using the same scaler
    input_test_scaled = scaler.transform(inputs_test)

    # 4️⃣ Fit the model on the scaled training data
    model = LinearRegression().fit(input_train_scaled, targets_train)

    # 5️⃣ Predict on the scaled test data
    prediction_test = model.predict(input_test_scaled)

    # 6️⃣ Evaluate
    loss = root_mean_squared_error(targets_test, prediction_test)
    print("After Data Split loss:", loss)


def Before_Data_Split(Dataframe):
    enc = preprocessing.OneHotEncoder()
    enc.fit(Dataframe[['region']])
    one_hot = enc.transform(Dataframe[['region']])
    one_hot = one_hot.toarray()

    Dataframe[['northeast', 'northwest', 'southeast', 'southwest']] = one_hot
    smoker_codes = {'no' : 0 , 'yes' : 1 }
    sex_codes = {'male' : 1 , 'female' : 0 }
    Dataframe['smoker_code'] = Dataframe['smoker'].map(smoker_codes)
    Dataframe['sex_code'] = Dataframe['sex'].map(sex_codes)

    numeric_cols = ['age','bmi','children']
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(Dataframe[numeric_cols])

    cat_cols = ['smoker_code', 'sex_code', 'northeast', 'northwest', 'southeast', 'southwest']
    categorical_data = Dataframe[cat_cols].values

    inputs = np.concatenate((scaled_data,categorical_data), axis=1)
    #inputs = np.concatenate((medical_df[numeric_cols], medical_df[cat_cols]), axis=1)
    targets = Dataframe['charges']

    inputs_train,inputs_test,targets_train,targets_test = train_test_split(inputs,targets,test_size=0.1)

    #inputs_train,inputs_test,targets_train,targets_test = train_test_split(inputs,targets,test_size=0.1)
    model = LinearRegression().fit(inputs_train,targets_train)
    prediction_test = model.predict(inputs_test)

    loss = root_mean_squared_error(targets_test, prediction_test)

    print("Before Data Split loss:",loss)

def main():
    medical_df = pd.read_csv('medical.csv')
    After_Data_Split(medical_df.copy())
    Before_Data_Split(medical_df.copy())

if __name__ == "__main__":
    main()


