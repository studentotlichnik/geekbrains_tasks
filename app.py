import flask
from flask import render_template

import pandas as pd
import xgboost as xgb

app = flask.Flask(__name__, template_folder='templates')

'''
прописываем путь, по которому будем работать с моделью
обзательно надо происать методы гет и пост
'''
@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])

'''
объявим функцию, которая выполняет методы
и что они должны выполнять
'''

def main():
    if flask.request.method == 'GET':
        return render_template('main.html' )
        
    if flask.request.method == 'POST':
        '''
        определяем переменные которые будут приходить 
        из гет запроса + в какой тип их определяем
        по условию в main.html прописал текст
        '''
        Exposure = float(flask.request.form['Exposure'])
        LicAge = int(flask.request.form['LicAge'])
        RecordBeg = flask.request.form['RecordBeg']
        RecordEnd = flask.request.form['RecordEnd']
        VehAge = flask.request.form['VehAge']
        Gender = flask.request.form['Gender']
        MariStat = flask.request.form['MariStat']
        SocioCateg = flask.request.form['SocioCateg']
        VehUsage = flask.request.form['VehUsage']
        DrivAge = int(flask.request.form['DrivAge'])
        HasKmLimit = int(flask.request.form['HasKmLimit'])
        BonusMalus = int(flask.request.form['BonusMalus'])
        VehBody = flask.request.form['VehBody']
        VehPrice = flask.request.form['VehPrice']
        VehEngine = flask.request.form['VehEngine']
        VehEnergy = flask.request.form['VehEnergy']
        VehMaxSpeed = flask.request.form['VehMaxSpeed']
        VehClass = flask.request.form['VehClass']
        RiskVar = float(flask.request.form['RiskVar'])
        Garage = flask.request.form['Garage']
        ClaimInd = int(flask.request.form['ClaimInd'])
        Dataset = int(flask.request.form['Dataset'])
        DeducType = flask.request.form['DeducType']
        ClaimNbResp = float(flask.request.form['ClaimNbResp'])
        ClaimNbNonResp = float(flask.request.form['ClaimNbNonResp'])
        ClaimNbParking = float(flask.request.form['ClaimNbParking'])
        ClaimNbWindscreen = float(flask.request.form['ClaimNbWindscreen'])
        OutUseNb = float(flask.request.form['OutUseNb'])
        RiskArea = float(flask.request.form['RiskArea'])

        '''
        формирую список из признаков, для окторых надо предсказать
        '''

        to_pred = [Exposure, LicAge, RecordBeg, RecordEnd, VehAge, Gender,
        MariStat, SocioCateg, VehUsage, DrivAge, HasKmLimit, BonusMalus,
        VehBody, VehPrice, VehEngine, VehEnergy, VehMaxSpeed, VehClass,
        RiskVar, Garage, ClaimInd, Dataset, DeducType, ClaimNbResp, 
        ClaimNbNonResp, ClaimNbParking, ClaimNbWindscreen, OutUseNb,
        RiskArea]
        
        '''
        формирую колонки с запроса гет и с теми, на которые обучилась моя модель
        можно сделать умнее, здесь ручной режим, чтобы ускориться
        '''

        clmns = ['Exposure', 'LicAge', 'RecordBeg', 'RecordEnd', 'VehAge', 
        'Gender', 'MariStat', 'SocioCateg', 'VehUsage', 'DrivAge', 
        'HasKmLimit', 'BonusMalus', 'VehBody', 'VehPrice', 'VehEngine', 
        'VehEnergy', 'VehMaxSpeed', 'VehClass','RiskVar', 'Garage', 
        'ClaimInd', 'Dataset', 'DeducType', 'ClaimNbResp', 'ClaimNbNonResp',
        'ClaimNbParking', 'ClaimNbWindscreen', 'OutUseNb','RiskArea']

        x_clmns = ['Exposure', 'LicAge', 'VehAge', 'Gender', 'MariStat', 'DrivAge',
        'HasKmLimit', 'BonusMalus', 'RiskVar', 'ClaimInd', 'Dataset',
        'ClaimNbResp', 'ClaimNbNonResp', 'ClaimNbParking', 'ClaimNbFireTheft',
        'ClaimNbWindscreen', 'OutUseNb', 'RiskArea', 'SocioCateg_CSP1',
        'SocioCateg_CSP2', 'SocioCateg_CSP3', 'SocioCateg_CSP4',
        'SocioCateg_CSP5', 'SocioCateg_CSP6', 'SocioCateg_CSP7',
        'SocioCateg_CSP8', 'SocioCateg_CSP9', 'VehUsage_Private',
        'VehUsage_Private+trip to office', 'VehUsage_Professional',
        'VehUsage_Professional run', 'VehBody_bus', 'VehBody_cabriolet',
        'VehBody_coupe', 'VehBody_microvan', 'VehBody_other microvan',
        'VehBody_sedan', 'VehBody_sport utility vehicle',
        'VehBody_station wagon', 'VehBody_van', 'VehEngine_GPL',
        'VehEngine_carburation', 'VehEngine_direct injection overpowered',
        'VehEngine_electric', 'VehEngine_injection',
        'VehEngine_injection overpowered', 'VehEnergy_GPL', 'VehEnergy_diesel',
        'VehEnergy_eletric', 'VehEnergy_regular', 'VehMaxSpeed_1-130 km/h',
        'VehMaxSpeed_130-140 km/h', 'VehMaxSpeed_140-150 km/h',
        'VehMaxSpeed_150-160 km/h', 'VehMaxSpeed_160-170 km/h',
        'VehMaxSpeed_170-180 km/h', 'VehMaxSpeed_180-190 km/h',
        'VehMaxSpeed_190-200 km/h', 'VehMaxSpeed_200-220 km/h',
        'VehMaxSpeed_220+ km/h', 'VehClass_0', 'VehClass_A', 'VehClass_B',
        'VehClass_H', 'VehClass_M1', 'VehClass_M2', 'Garage_Collective garage',
        'Garage_None', 'Garage_Private garage']

        '''
        тривиальная функция которая обрабатывает признаки 
        '''
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

        '''
        в этом куске кода я забираю переменные от пользователя
        от метода гет
        после чего обрабатываю признаки по аналогии с тем,
        как учил модель
        '''
        to_pred = pd.DataFrame([to_pred], columns=clmns)

        test = fast_preproc(to_pred)

        for i in [clm for clm in x_clmns if clm not in test.columns]:
            test[i] = 0
        
        '''
        искджибуст требует особый формат данных, который и определяю
        '''
        test = xgb.DMatrix(test.values)

        '''
        вызываю обученную и сохраненную ранее модель
        формат любой подходящий
        для буста стабильнее json
        '''

        model = xgb.Booster(model_file='model.model')

        '''
        предсказываю
        '''
        y_pred = model.predict(test)

        '''
        отправляю результат через метод пост
        обратно в html
        '''
        return render_template('main.html', result = y_pred)

if __name__ == '__main__':
    app.run()