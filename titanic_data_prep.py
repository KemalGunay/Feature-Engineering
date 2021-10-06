# LIBRARIES
import pandas as pd
from helpers.eda import check_df, grab_col_names
from helpers.data_prep import check_outlier, outlier_thresholds, grab_outliers, \
    remove_outlier, replace_with_thresholds, missing_values_table, label_encoder, \
    rare_encoder, rare_analyser, one_hot_encoder
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler


# LOAD DATA
def load_titanic():
    data = pd.read_csv("titanic.csv")
    return data

df = load_titanic()
df.head()


# titanic_data_pref FUNCTION
def titanic_data_prep(dataframe):
    check_df(dataframe)
    dataframe.columns = [col.upper() for col in dataframe.columns]
# Feature interactions/engineering
# Cabin bool
    dataframe["NEW_CABIN_BOOL"] = dataframe["CABIN"].notnull().astype('int')
# Name count
    dataframe["NEW_NAME_COUNT"] = dataframe["NAME"].str.len()
# name word count
    dataframe["NEW_NAME_WORD_COUNT"] = dataframe["NAME"].apply(lambda x: len(str(x).split(" ")))
# name dr
    dataframe["NEW_NAME_DR"] = dataframe["NAME"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
# name title
    dataframe['NEW_TITLE'] = dataframe.NAME.str.extract(' ([A-Za-z]+)\.', expand=False)
# family size: ailes sayısı
    dataframe["NEW_FAMILY_SIZE"] = dataframe["SIBSP"] + dataframe["PARCH"] + 1
# age_pclass:
    dataframe["NEW_AGE_PCLASS"] = dataframe["AGE"] * dataframe["PCLASS"]
# is alone
    dataframe.loc[((dataframe['SIBSP'] + dataframe['PARCH']) > 0), "NEW_IS_ALONE"] = "NO"
    dataframe.loc[((dataframe['SIBSP'] + dataframe['PARCH']) == 0), "NEW_IS_ALONE"] = "YES"
# age level:
    dataframe.loc[(dataframe['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
    dataframe.loc[(dataframe['AGE'] >= 18) & (dataframe['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
    dataframe.loc[(dataframe['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'
# sex age :
    dataframe.loc[(dataframe['SEX'] == 'male') & (dataframe['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
    dataframe.loc[(dataframe['SEX'] == 'male') & ((dataframe['AGE'] > 21) & (dataframe['AGE']) <= 50), 'NEW_SEX_CAT'] = 'maturemale'
    dataframe.loc[(dataframe['SEX'] == 'male') & (dataframe['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
    dataframe.loc[(dataframe['SEX'] == 'female') & (dataframe['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
    dataframe.loc[(dataframe['SEX'] == 'female') & ((dataframe['AGE'] > 21) & (dataframe['AGE']) <= 50), 'NEW_SEX_CAT'] = 'maturefemale'
    dataframe.loc[(dataframe['SEX'] == 'female') & (dataframe['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'

    # grab_col_names
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)

    # passengerid not neccessary, remove it
    num_cols = [col for col in num_cols if "PASSENGERID" not in col]


    # CHECK OUTLIERS
    for col in num_cols:
        print(col, check_outlier(dataframe, col))
    # replace outliers
    for col in num_cols:
        replace_with_thresholds(dataframe, col)
    # check outlier
    for col in num_cols:
        print(col, check_outlier(dataframe, col))


    # MISSING VALUES
    missing_values_table(dataframe)
    dataframe.drop("CABIN", inplace=True, axis=1)

    remove_cols = ["TICKET", "NAME"]
    dataframe.drop(remove_cols, inplace=True, axis=1)

    dataframe["AGE"] = dataframe["AGE"].fillna(dataframe.groupby("NEW_TITLE")["AGE"].transform("median"))
    dataframe["NEW_AGE_PCLASS"] = dataframe["AGE"] * dataframe["PCLASS"]
    # yaş gitse bile yaşa bağlı değişkenlerde eksiklikler devam ediyor. Bu yüzden yaşa bağlı değişkenleri bir daha oluşturuyoruö
    dataframe.loc[(dataframe['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
    dataframe.loc[(dataframe['AGE'] >= 18) & (dataframe['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
    dataframe.loc[(dataframe['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'
    dataframe.loc[(dataframe['SEX'] == 'male') & (dataframe['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
    dataframe.loc[(dataframe['SEX'] == 'male') & (
                (dataframe['AGE'] > 21) & (dataframe['AGE']) <= 50), 'NEW_SEX_CAT'] = 'maturemale'
    dataframe.loc[(dataframe['SEX'] == 'male') & (dataframe['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
    dataframe.loc[(dataframe['SEX'] == 'female') & (dataframe['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
    dataframe.loc[(dataframe['SEX'] == 'female') & (
                (dataframe['AGE'] > 21) & (dataframe['AGE']) <= 50), 'NEW_SEX_CAT'] = 'maturefemale'
    dataframe.loc[(dataframe['SEX'] == 'female') & (dataframe['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'

    dataframe = dataframe.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)

    # LABEL ENCODING
    binary_cols = [col for col in dataframe.columns if
                   dataframe[col].dtype not in [int, float] and dataframe[col].nunique() == 2]

    for col in binary_cols:
        dataframe = label_encoder(dataframe, col)

    # RARE ANALYSER
    rare_analyser(dataframe, "SURVIVED", cat_cols)
    dataframe = rare_encoder(dataframe, 0.01, cat_cols)

    # ONE-HOT ENCODING
    ohe_cols = [col for col in dataframe.columns if 10 >= dataframe[col].nunique() > 2]
    dataframe = one_hot_encoder(dataframe, ohe_cols)
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)
    num_cols = [col for col in num_cols if "PASSENGERID" not in col]

    rare_analyser(dataframe, "SURVIVED", cat_cols)
    useless_cols = [col for col in dataframe.columns if dataframe[col].nunique() == 2 and
                    (dataframe[col].value_counts() / len(dataframe) < 0.01).any(axis=None)]
    dataframe.drop(useless_cols, axis=1, inplace=True)
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)
    num_cols = [col for col in num_cols if col not in "PassengerId"]
    dataframe = pd.get_dummies(dataframe[cat_cols + num_cols], drop_first=True)
    # ROBUST SCALER
    rs = RobustScaler()
    dataframe = pd.DataFrame(rs.fit_transform(dataframe), columns=dataframe.columns)

    return dataframe

prep_df = titanic_data_prep(df)
check_df(prep_df)

# # Save the preprocessed data set to disk with pickle.
prep_df.to_pickle("./titanic_data_prep.pkl")
pd.read_pickle("titanic.pkl")
