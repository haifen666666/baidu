import pandas as pd
'''
wjy_train = pd.read_csv('wjy_train_feature.csv')
wjy_train.sort_values('area_id',inplace=True)
wjy_train.reset_index(drop=True,inplace=True)
wjy_train = wjy_train[['day_mean_num','u_day_mean_num','weekend_num_mean','weekday_num_mean','u_weekend_num_mean',
                      'u_weekday_num_mean','mg_num_mean','after_num_mean','eve_num_mean','night_num_mean',
                      'u_mg_num_mean','u_after_num_mean','u_eve_num_mean','u_night_num_mean','area_id']]
train = pd.read_csv('train_feature3.csv')
train = wjy_train.merge(train,on='area_id',how='inner')
print(train.shape)
train.to_csv('train_feature4.csv',index=False)

wjy_test = pd.read_csv('wjy_test_feature.csv')
wjy_test.sort_values('area_id',inplace=True)
wjy_test.reset_index(drop=True,inplace=True)
wjy_test = wjy_test[['day_mean_num','u_day_mean_num','weekend_num_mean','weekday_num_mean','u_weekend_num_mean',
                      'u_weekday_num_mean','mg_num_mean','after_num_mean','eve_num_mean','night_num_mean',
                      'u_mg_num_mean','u_after_num_mean','u_eve_num_mean','u_night_num_mean','area_id']]
test = pd.read_csv('test_feature3.csv')
test = wjy_test.merge(test,on='area_id',how='inner')
print(test.shape)
test.to_csv('test_feature4.csv',index=False)
'''

'''
train = pd.read_csv('train_feature3.csv')
train2 = pd.read_csv('wjy_train_feature.csv')
train3 = train2.merge(train,on='area_id',how='inner')
train3['mean_A0_day0'] = train3['meanA0'] / (train3['day_mean0'] + 0.0001)
train3['mean_A1_day1'] = train3['meanA1'] / (train3['day_mean1'] + 0.0001)
train3['mean_A2_day2'] = train3['meanA2'] / (train3['day_mean2'] + 0.0001)
train3['mean_A3_day3'] = train3['meanA3'] / (train3['day_mean3'] + 0.0001)
train3['mean_A4_day4'] = train3['meanA4'] / (train3['day_mean4'] + 0.0001)
train3['mean_A5_day5'] = train3['meanA5'] / (train3['day_mean5'] + 0.0001)
train3['mean_A6_day6'] = train3['meanA6'] / (train3['day_mean6'] + 0.0001)
train3['mean_A7_day7'] = train3['meanA7'] / (train3['day_mean7'] + 0.0001)
train3.to_csv('train_feature4.csv',index=False)
'''

train = pd.read_csv('test_feature3.csv')
train2 = pd.read_csv('wjy_test_feature.csv')
train3 = train2.merge(train,on='area_id',how='inner')
train3['mean_A0_day0'] = train3['meanA0'] / (train3['day_mean0'] + 0.0001)
train3['mean_A1_day1'] = train3['meanA1'] / (train3['day_mean1'] + 0.0001)
train3['mean_A2_day2'] = train3['meanA2'] / (train3['day_mean2'] + 0.0001)
train3['mean_A3_day3'] = train3['meanA3'] / (train3['day_mean3'] + 0.0001)
train3['mean_A4_day4'] = train3['meanA4'] / (train3['day_mean4'] + 0.0001)
train3['mean_A5_day5'] = train3['meanA5'] / (train3['day_mean5'] + 0.0001)
train3['mean_A6_day6'] = train3['meanA6'] / (train3['day_mean6'] + 0.0001)
train3['mean_A7_day7'] = train3['meanA7'] / (train3['day_mean7'] + 0.0001)
train3.to_csv('test_feature4.csv',index=False)

