import pandas as pd
column =     ['maxA0','minA0','meanA0','medianA0','stdA0','varA0','maxB0','minB0','meanB0','medianB0','stdB0','varB0',
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

              'area_id','answer']
def make_new_train():
    train = pd.read_csv('test_feature.csv')  #这里没用test命名   是因为这是直接根据训练集合拿来的代码 后面全是train命名
    '''
    new_columns = train.columns.drop(['minA0','minA1','minA2','minA3','minA4','minA5','minA6','minA7',
                                      'minB0','minB1','minB2','minB3','minB4','minB5','minB6','minB7',
                                      'minC0','minC1','minC2','minC3','minC4','minC5','minC6','minC7',
                                      'minD0','minD1','minD2','minD3','minD4','minD5','minD6','minD7',
                                      'minE0','minE1','minE2','minE3','minE4','minE5','minE6','minE7'])
                                      
    train = train[new_columns]
    '''
    train['meanBA0'] = train['meanB0'] / (train['meanA0']+0.0001)  #这里0.0001是为了避免分母为0
    train['meanCA0'] = train['meanC0'] / (train['meanA0']+0.0001)
    train['meanDA0'] = train['meanD0'] / (train['meanA0']+0.0001)
    train['meanEA0'] = train['meanE0'] / (train['meanA0']+0.0001)

    train['meanBA1'] = train['meanB1'] / (train['meanA1']+0.0001)  #这里0.0001是为了避免分母为0
    train['meanCA1'] = train['meanC1'] / (train['meanA1']+0.0001)
    train['meanDA1'] = train['meanD1'] / (train['meanA1']+0.0001)
    train['meanEA1'] = train['meanE1'] / (train['meanA1']+0.0001)

    train['meanBA2'] = train['meanB2'] / (train['meanA2']+0.0001)  #这里0.0001是为了避免分母为0
    train['meanCA2'] = train['meanC2'] / (train['meanA2']+0.0001)
    train['meanDA2'] = train['meanD2'] / (train['meanA2']+0.0001)
    train['meanEA2'] = train['meanE2'] / (train['meanA2']+0.0001)

    train['meanBA3'] = train['meanB3'] / (train['meanA3']+0.0001)  #这里0.0001是为了避免分母为0
    train['meanCA3'] = train['meanC3'] / (train['meanA3']+0.0001)
    train['meanDA3'] = train['meanD3'] / (train['meanA3']+0.0001)
    train['meanEA3'] = train['meanE3'] / (train['meanA3']+0.0001)

    train['meanBA5'] = train['meanB5'] / (train['meanA5']+0.0001)  #这里0.0001是为了避免分母为0
    train['meanCA5'] = train['meanC5'] / (train['meanA5']+0.0001)
    train['meanDA5'] = train['meanD5'] / (train['meanA5']+0.0001)
    train['meanEA5'] = train['meanE5'] / (train['meanA5']+0.0001)

    train['meanBA6'] = train['meanB6'] / (train['meanA6']+0.0001)  #这里0.0001是为了避免分母为0
    train['meanCA6'] = train['meanC6'] / (train['meanA6']+0.0001)
    train['meanDA6'] = train['meanD6'] / (train['meanA6']+0.0001)
    train['meanEA6'] = train['meanE6'] / (train['meanA6']+0.0001)

    train['meanBA7'] = train['meanB7'] / (train['meanA7']+0.0001)  #这里0.0001是为了避免分母为0
    train['meanCA7'] = train['meanC7'] / (train['meanA7']+0.0001)
    train['meanDA7'] = train['meanD7'] / (train['meanA7']+0.0001)
    train['meanEA7'] = train['meanE7'] / (train['meanA7']+0.0001)

    train['percent_date_times_0'] = train['date_times_0'] / 116
    train['percent_date_times_1'] = train['date_times_0'] / 46
    train['percent_date_times_2'] = train['date_times_0'] / 17
    train['percent_date_times_4'] = train['date_times_0'] / 182

    train['meanA01'] = train['meanA0'] / (train['meanA1']+0.0001)
    train['meanB01'] = train['meanB0'] / (train['meanB1']+0.0001)
    train['meanC01'] = train['meanC0'] / (train['meanC1']+0.0001)
    train['meanD01'] = train['meanD0'] / (train['meanD1']+0.0001)
    train['meanE01'] = train['meanE0'] / (train['meanE1']+0.0001)

    train['meanA02'] = train['meanA0'] / (train['meanA2']+0.0001)
    train['meanB02'] = train['meanB0'] / (train['meanB2']+0.0001)
    train['meanC02'] = train['meanC0'] / (train['meanC2']+0.0001)
    train['meanD02'] = train['meanD0'] / (train['meanD2']+0.0001)
    train['meanE02'] = train['meanE0'] / (train['meanE2']+0.0001)

    train['meanA03'] = train['meanA0'] / (train['meanA3']+0.0001)
    train['meanB03'] = train['meanB0'] / (train['meanB3']+0.0001)
    train['meanC03'] = train['meanC0'] / (train['meanC3']+0.0001)
    train['meanD03'] = train['meanD0'] / (train['meanD3']+0.0001)
    train['meanE03'] = train['meanE0'] / (train['meanE3']+0.0001)

    train['meanA13'] = train['meanA1'] / (train['meanA3']+0.0001)
    train['meanB13'] = train['meanB1'] / (train['meanB3']+0.0001)
    train['meanC13'] = train['meanC1'] / (train['meanC3']+0.0001)
    train['meanD13'] = train['meanD1'] / (train['meanD3']+0.0001)
    train['meanE13'] = train['meanE1'] / (train['meanE3']+0.0001)

    train['meanA05'] = train['meanA0'] / (train['meanA5']+0.0001)
    train['meanB05'] = train['meanB0'] / (train['meanB5']+0.0001)
    train['meanC05'] = train['meanC0'] / (train['meanC5']+0.0001)
    train['meanD05'] = train['meanD0'] / (train['meanD5']+0.0001)
    train['meanE05'] = train['meanE0'] / (train['meanE5']+0.0001)

    train['meanA06'] = train['meanA0'] / (train['meanA6']+0.0001)
    train['meanB06'] = train['meanB0'] / (train['meanB6']+0.0001)
    train['meanC06'] = train['meanC0'] / (train['meanC6']+0.0001)
    train['meanD06'] = train['meanD0'] / (train['meanD6']+0.0001)
    train['meanE06'] = train['meanE0'] / (train['meanE6']+0.0001)

    train['meanA07'] = train['meanA0'] / (train['meanA7']+0.0001)
    train['meanB07'] = train['meanB0'] / (train['meanB7']+0.0001)
    train['meanC07'] = train['meanC0'] / (train['meanC7']+0.0001)
    train['meanD07'] = train['meanD0'] / (train['meanD7']+0.0001)
    train['meanE07'] = train['meanE0'] / (train['meanE7']+0.0001)

    train.to_csv('test_feature2.csv',index=False)

make_new_train()



