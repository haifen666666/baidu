import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


def train(filename):
    train = pd.read_csv(filename)
    train['answer'] = train['answer'] - 1

    tt_valid = pd.read_csv('tt_val.csv')
    tt_list = list(tt_valid['Id'])
    train_new = train[~train['area_id'].isin(tt_list)]
    test_new = train[train['area_id'].isin(tt_list)]

    del train_new['area_id']
    del train_new['label']
    y_train = train_new[['answer']]
    del train_new['answer']
    x_train = train_new


    valid = test_new[['area_id']]
    del test_new['area_id']
    del test_new['label']
    y_test = test_new[['answer']]
    del test_new['answer']
    x_test = test_new

    answer = pd.concat([y_train,y_test],ignore_index=True)
    train = pd.concat([x_train,x_test],ignore_index=True)
    print(answer.shape,train.shape)




    lgb_train = lgb.Dataset(x_train, y_train, free_raw_data=False)
    lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train, free_raw_data=False)
    lgb_all = lgb.Dataset(train,answer,free_raw_data=False)
    params = {'boosting_type': 'gbdt',
              'objective': 'multiclass',
              'metrics': 'multi_logloss',
              'nthread': 10,   #线程数
              'num_class': 9,  #类别数
              'learning_rate': 0.02,
              'num_leaves': 150,
              'max_depth': 16,
              'max_bin': 200, #将feature存入bin的最大值，越大越准，最大255,默认值255
              'subsample_for_bin': 50000, #用于构建直方图数据的数量，默认值为20000,越大训练效果越好，但速度会越慢
              'subsample': 0.8, #子采样，为了防止过拟合
              'subsample_freq': 1,  #重采样频率,如果为正整数，表示每隔多少次迭代进行bagging
              'colsample_bytree': 0.8, #每棵随机采样的列数的占比,一般取0.5-1
              'reg_alpha': 0.2, #L1正则化项，越大越保守
              'reg_lambda': 0, #L2正则化项，越大越保守
              'min_split_gain': 0.0,
              'min_child_weight': 1, #默认值为1,越大越能避免过拟合，建议使用CV调整
              'min_child_samples': 10, #alias：min_data_in_leaf 越大越能避免树过深，避免过拟合，但是可能欠拟合 需要CV调整
              'scale_pos_weight': 1, # 类别不均衡时设定,
              }
    num_round = 3000
    
    model_train = lgb.train(params,
                    lgb_train,
                    num_round,
                    categorical_feature=['top1_0','top2_0','top3_0','top1_1','top2_1','top3_1',
                                         'top1_2','top2_2','top3_2','top1_3','top2_3','top3_3',
                                         'top1_4','top2_4','top3_4','top1_5','top2_5','top3_5',
                                         'top1_6','top2_6','top3_6','top1_7','top2_7','top3_7'],
                    valid_sets=lgb_eval, early_stopping_rounds = 100)
    #线下验证正确率
    pred =model_train.predict(x_test)
    pred = [list(x).index(max(x)) for x in pred]
    actually = list(y_test['answer'])
    observe = pd.DataFrame()
    observe['Id'] = valid['area_id']
    observe['pred'] = pred
    observe['actually'] = actually
    observe['equal'] = observe['pred'] - observe['actually']
    observe.to_csv('zyn_observe.csv',index=False)
    right = observe[observe['equal'] == 0].shape[0]
    all = observe.shape[0]
    print(right/all)
    print(model_train.best_iteration)
    '''
    model = lgb.train(params, lgb_all, model_train.best_iteration,
                     categorical_feature=['top1_0','top2_0','top3_0','top1_1','top2_1','top3_1',
                                         'top1_2','top2_2','top3_2','top1_3','top2_3','top3_3',
                                         'top1_4','top2_4','top3_4','top1_5','top2_5','top3_5',
                                         'top1_6','top2_6','top3_6','top1_7','top2_7','top3_7'])
    model.save_model('lgb1.model') # 用于存储训练出的模型
    dfFeature = pd.DataFrame()
    dfFeature['featureName'] = model.feature_name()
    dfFeature['score'] = model.feature_importance()
    dfFeature.to_csv('featureImportance1.csv')
    '''



def run_lgb():
    column = ['maxA0','minA0','meanA0','medianA0','stdA0','varA0','maxB0','minB0','meanB0','medianB0','stdB0','varB0',
              'maxC0','minC0','meanC0','medianC0','stdC0','varC0','maxD0','minD0','meanD0','medianD0','stdD0','varD0',
              'maxE0','minE0','meanE0','medianE0','stdE0','varE0','top1_0','top2_0','top3_0','date_times_0',

              'maxA1','minA1','meanA1','medianA1','stdA1','varA1','maxB1','minB1','meanB1','medianB1','stdB1','varB1',
              'maxC1','minC1','meanC1','medianC1','stdC1','varC1','maxD1','minD1','meanD1','medianD1','stdD1','varD1',
              'maxE1','minE1','meanE1','medianE1','stdE1','varE1','top1_1','top2_1','top3_1','date_times_1',

              'maxA2','minA2','meanA2','medianA2','stdA2','varA2','maxB2','minB2','meanB2','medianB2','stdB2','varB2',
              'maxC2','minC2','meanC2','medianC2','stdC2','varC2','maxD2','minD2','meanD2','medianD2','stdD2','varD2',
              'maxE2','minE2','meanE2','medianE2','stdE2','varE2','top1_2','top2_2','top3_2','date_times_2',

              'maxA3','minA3','meanA3','medianA3','stdA3','varA3','maxB3','minB3','meanB3','medianB3','stdB3','varB3',
              'maxC3','minC3','meanC3','medianC3','stdC3','varC3','maxD3','minD3','meanD3','medianD3','stdD3','varD3',
              'maxE3','minE3','meanE3','medianE3','stdE3','varE3','top1_3','top2_3','top3_3','date_times_3',

              'maxA4','minA4','meanA4','medianA4','stdA4','varA4','maxB4','minB4','meanB4','medianB4','stdB4','varB4',
              'maxC4','minC4','meanC4','medianC4','stdC4','varC4','maxD4','minD4','meanD4','medianD4','stdD4','varD4',
              'maxE4','minE4','meanE4','medianE4','stdE4','varE4','top1_4','top2_4','top3_4','date_times_4',

              'maxA5','minA5','meanA5','medianA5','stdA5','varA5','maxB5','minB5','meanB5','medianB5','stdB5','varB5',
              'maxC5','minC5','meanC5','medianC5','stdC5','varC5','maxD5','minD5','meanD5','medianD5','stdD5','varD5',
              'maxE5','minE5','meanE5','medianE5','stdE5','varE5','top1_5','top2_5','top3_5','date_times_5',

              'maxA6','minA6','meanA6','medianA6','stdA6','varA6','maxB6','minB6','meanB6','medianB6','stdB6','varB6',
              'maxC6','minC6','meanC6','medianC6','stdC6','varC6','maxD6','minD6','meanD6','medianD6','stdD6','varD6',
              'maxE6','minE6','meanE6','medianE6','stdE6','varE6','top1_6','top2_6','top3_6','date_times_6',

              'maxA7','minA7','meanA7','medianA7','stdA7','varA7','maxB7','minB7','meanB7','medianB7','stdB7','varB7',
              'maxC7','minC7','meanC7','medianC7','stdC7','varC7','maxD7','minD7','meanD7','medianD7','stdD7','varD7',
              'maxE7','minE7','meanE7','medianE7','stdE7','varE7','top1_7','top2_7','top3_7','date_times_7',
              'meanBA0','meanCA0','meanDA0','meanEA0','meanBA1','meanCA1','meanDA1','meanEA1','meanBA2','meanCA2','meanDA2',
              'meanEA2','meanBA3','meanCA3','meanDA3','meanEA3','meanBA5','meanCA5','meanDA5','meanEA5','meanBA6','meanCA6',
              'meanDA6','meanEA6','meanBA7','meanCA7','meanDA7','meanEA7','percent_date_times_0','percent_date_times_1',
              'percent_date_times_2','percent_date_times_4','meanA01','meanB01','meanC01','meanD01','meanE01','meanA02',
              'meanB02','meanC02','meanD02','meanE02','meanA03','meanB03','meanC03','meanD03','meanE03','meanA13','meanB13',
              'meanC13','meanD13','meanE13','meanA05','meanB05','meanC05','meanD05','meanE05','meanA06','meanB06','meanC06',
              'meanD06','meanE06','meanA07','meanB07','meanC07','meanD07','meanE07','area_id','answer']

    train('train_feature6.csv')

def predict():
    model = lgb.Booster(model_file = 'lgb1.model') #init model
    test = pd.read_csv('test_feature6.csv')
    test.sort_values('area_id',inplace=True)

    submit = test[['area_id']]
    del test['area_id']
    pred = model.predict(test)

    #得到预测的排序结果，保存下来，用于做结果层面的特征融合
    lgb_sort = pd.DataFrame(pred,columns=['1','2','3','4','5','6','7','8','9'])
    lgb_sort.to_csv('lgb_sort.csv',index=False)

    #选出概率最大的一个，拿到对应label，作为类标号
    pred = [list(x).index(max(x)) for x in pred]
    submit['answer'] = pred

    #按照提交格式，处理数据后保存
    submit['answer'] = submit['answer'] + 1
    submit['area_id'] = submit['area_id'].astype('str')
    submit['answer'] = submit['answer'].astype('str')

    def fill_0(x):
        tt = (6 - len(x.area_id))* '0'
        x.area_id = tt + x.area_id
        dd = (3 - len(x.answer)) * '0'
        x.answer = dd + x.answer
        return x

    submit = submit.apply(fill_0,axis=1)
    submit.to_csv('observe.csv',index=False)  #为了便于查看答案的分布
    submit.to_csv('submit.txt',sep='\t',index=None,header=None)


if __name__ == '__main__':
    run_lgb()
    #predict()





