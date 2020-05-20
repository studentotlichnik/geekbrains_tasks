import pandas as pd
import xgboost as xgb


df = pd.read_csv('freMPL-R.csv', low_memory=False)

df.loc[df.ClaimAmount < 0, 'ClaimAmount'] = 0

def fast_preproc(df):
    # encoding object types (by meaning valid for binaries)
    def SeriesFactorizer(series):
        series, unique = pd.factorize(series)
        reference = {x: i for x, i in enumerate(unique)}
        return series, reference

    # encoding gender
    df.Gender, GenderRef = SeriesFactorizer(df.Gender)
    # encdoing MariStat
    df.MariStat, MariStatRef = SeriesFactorizer(df.MariStat)
    # decreasing the depth of the features 
    df.SocioCateg = df.SocioCateg.str.slice(0,4)
    # encoding VehAge (order makes sense)
    df.VehAge.replace(
        {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6-7': 6, '8-9': 7, '10+': 8},
        inplace=True
    )
    df.VehAge.fillna(9, inplace=True)

    # converting all other that makes sense to dummies
    clms_to_dummies = ['SocioCateg', 'VehUsage', 'VehBody', 'VehEngine', 'VehEnergy', 
                    'VehMaxSpeed', 'VehClass', 'Garage']

    tbd = pd.DataFrame(index=range(df.shape[0]),columns=['tbd'])

    for clm in clms_to_dummies:
        tbd = tbd.join(pd.get_dummies(df[clm], prefix=clm))

    df = df.join(tbd)
    df.drop(clms_to_dummies + ['tbd'], axis=1, inplace=True)
    
    # dropping the object that are left
    df = df.select_dtypes(exclude=['object'])

    return df

df = fast_preproc(df)
y = df.ClaimAmount
x_clmns = df.drop('ClaimAmount', axis=1).columns

x = df.drop('ClaimAmount', axis=1).values

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

model = xgb.XGBRegressor(
    # colsample_bytree=0.5, gamma=0.0, 
    # learning_rate=0.005, max_depth=7, 
    # min_child_weight=0.5, n_estimators=5800,
    # reg_alpha=0.9, reg_lambda=0.99,
    # subsample=0.99,seed=42, silent=1,
)

model.fit(x_scaled,y)

model.save_model('model1.model')

# model = xgb.Booster(model_file='model.model')

# x_clmns = ['Exposure', 'LicAge', 'VehAge', 'Gender', 'MariStat', 'DrivAge',
#             'HasKmLimit', 'BonusMalus', 'RiskVar', 'ClaimInd', 'Dataset',
#             'ClaimNbResp', 'ClaimNbNonResp', 'ClaimNbParking', 'ClaimNbFireTheft',
#             'ClaimNbWindscreen', 'OutUseNb', 'RiskArea', 'SocioCateg_CSP1',
#             'SocioCateg_CSP2', 'SocioCateg_CSP3', 'SocioCateg_CSP4',
#             'SocioCateg_CSP5', 'SocioCateg_CSP6', 'SocioCateg_CSP7',
#             'SocioCateg_CSP8', 'SocioCateg_CSP9', 'VehUsage_Private',
#             'VehUsage_Private+trip to office', 'VehUsage_Professional',
#             'VehUsage_Professional run', 'VehBody_bus', 'VehBody_cabriolet',
#             'VehBody_coupe', 'VehBody_microvan', 'VehBody_other microvan',
#             'VehBody_sedan', 'VehBody_sport utility vehicle',
#             'VehBody_station wagon', 'VehBody_van', 'VehEngine_GPL',
#             'VehEngine_carburation', 'VehEngine_direct injection overpowered',
#             'VehEngine_electric', 'VehEngine_injection',
#             'VehEngine_injection overpowered', 'VehEnergy_GPL', 'VehEnergy_diesel',
#             'VehEnergy_eletric', 'VehEnergy_regular', 'VehMaxSpeed_1-130 km/h',
#             'VehMaxSpeed_130-140 km/h', 'VehMaxSpeed_140-150 km/h',
#             'VehMaxSpeed_150-160 km/h', 'VehMaxSpeed_160-170 km/h',
#             'VehMaxSpeed_170-180 km/h', 'VehMaxSpeed_180-190 km/h',
#             'VehMaxSpeed_190-200 km/h', 'VehMaxSpeed_200-220 km/h',
#             'VehMaxSpeed_220+ km/h', 'VehClass_0', 'VehClass_A', 'VehClass_B',
#             'VehClass_H', 'VehClass_M1', 'VehClass_M2', 'Garage_Collective garage',
#             'Garage_None', 'Garage_Private garage']

# clmns = ['Exposure', 'LicAge', 'RecordBeg', 'RecordEnd', 'VehAge', 
# 'Gender', 'MariStat', 'SocioCateg', 'VehUsage', 'DrivAge', 
# 'HasKmLimit', 'BonusMalus', 'VehBody', 'VehPrice', 'VehEngine', 
# 'VehEnergy', 'VehMaxSpeed', 'VehClass','RiskVar', 'Garage', 'ClaimInd', 
# 'Dataset', 'DeducType', 'ClaimNbResp', 'ClaimNbNonResp', 'ClaimNbParking', 
# 'ClaimNbFireTheft', 'ClaimNbWindscreen', 'OutUseNb','RiskArea']


# to_pred = [0.441275, 309, '2004-08-16', '', '1', 'Female', 'Other', 
# 'CSP1', 'Professional', 34, 0, 63, 'other microvan', 'L', 
# 'injection', 'diesel',  '160-170 km/h', 'B', 15, 'Private garage', 
# 0, 1, '', '', '', '', '', '', '', '']

# to_pred = pd.DataFrame([to_pred], columns=clmns)

# test = fast_preproc(to_pred)

# for i in [clm for clm in x_clmns if clm not in test.columns]:
#     test[i] = 0

# test = test.values

# test_scaled = scaler.transform(test)

# y_pred = model.predict(xgb.DMatrix(test))

# print(y_pred)  

