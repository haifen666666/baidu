import pandas as pd
import numpy as np
from datetime import datetime

#对访问时间贴上时间标签，区分平时、周末、法定假日;上午、下午、晚上、深夜
def put_time_tag():
    visit = pd.read_csv('visit_hour_test.csv')
    visit['date'] = visit['year'] * 10000 + visit['month'] * 100 + visit['day']
    visit['date'] = visit['date'].astype('str')
    visit['week'] = visit['date'].apply(lambda x: datetime.strftime(datetime.strptime(x, '%Y%m%d'), '%w'))#需要大概10分钟
    visit['week'] = visit['week'].astype('int')
    visit['date'] = visit['date'].astype('int')

    visit.loc[(visit['hour'] <= 12) & (visit['hour'] >=6),'time_frame'] = 1
    visit.loc[(visit['hour'] <= 18) & (visit['hour'] >=12),'time_frame'] =2
    visit.loc[(visit['hour'] <= 24) & (visit['hour'] >=18),'time_frame'] =3
    visit.loc[(visit['hour'] <= 6) & (visit['hour'] >=0),'time_frame'] =4

    visit['date_cate'] = [0] * visit.shape[0]
    festival = [20181001,20181002,20181003,20181004,20181005,20181006,20181007,
                20181230,20181231,20190101,20190204,20190205,20190206,20190207,
                20190208,20190209,20190210]
    festival2 = [20181225,20190128,20190219] #分别是圣诞节，小年，元宵节 不放假的节日
    visit.loc[(visit['week']==0) | (visit['week']==6),'date_cate'] = 1
    visit.loc[visit['date'].isin(festival),'date_cate'] = 2
    visit.loc[visit['date'] == 20181229,'date_cate'] = 0 #这天虽然周六 但是调休
    visit.loc[visit['date'].isin(festival2),'date_cate'] = 3

    del visit['year']
    del visit['month']
    del visit['day']
    visit['time_frame'] = visit['time_frame'].astype('int')
    visit.to_hdf('visit_hour_test_new.h5',mode='w',key='tt',complevel=6,complib='blosc') #这里需要区分两次的文件，然后在终端合并

#获取统计特征，返回list
def get_statistic_features(df,theory_max) ->list:
    nums_list = list(df['nums'])
    length = len(nums_list)
    #如果理论上应该有7天的数据，实际采样只有5天有，则相差的两天手动补0
    if length != theory_max:
        differ = theory_max - length
        nums_list.extend([0]*differ)
    nums_array = np.array(nums_list)

    max = np.max(nums_array)
    min = np.min(nums_array)
    mean = np.mean(nums_array)
    median = np.median(nums_array)
    std = np.std(nums_array)
    var = np.var(nums_array)
    return [max,min,mean,median,std,var]

#按天统计特征,将每个小时人数相加
def get_day_info(df):
    df2 = df.groupby('date',as_index=False)['numbers'].agg({'nums':'sum'})
    return df2

def get_morning_info(df):
    df2 = df[df['time_frame'] == 1]
    df3 = df2.groupby('date',as_index=False)['numbers'].agg({'nums':'sum'})
    return df3

def get_afternoon_info(df):
    df2 = df[df['time_frame'] == 2]
    df3 = df2.groupby('date',as_index=False)['numbers'].agg({'nums':'sum'})
    return df3

def get_evening_info(df):
    df2 = df[df['time_frame'] == 3]
    df3 = df2.groupby('date',as_index=False)['numbers'].agg({'nums':'sum'})
    return df3

def get_night_info(df):
    df2 = df[df['time_frame'] == 4]
    df3 = df2.groupby('date',as_index=False)['numbers'].agg({'nums':'sum'})
    return df3

def get_top_hours(df):
    df2 = df.groupby('hour',as_index=False)['numbers'].agg({'nums':'sum'})
    hours = list(df2['hour'])
    nums = list(df2['nums'])
    if len(hours) < 3:
        return [np.nan,np.nan,np.nan]
    else:
        top1_loc = nums.index(max(nums))
        top1 = hours[top1_loc]
        nums.pop(top1_loc)
        hours.pop(top1_loc)

        top2_loc = nums.index(max(nums))
        top2 = hours[top2_loc]
        nums.pop(top2_loc)
        hours.pop(top2_loc)

        top3_loc = nums.index(max(nums))
        top3 = hours[top3_loc]
        return [top1,top2,top3]


#全天、上午、下午...不同时段的统计特征+全天最活跃3小时 最不活跃三小时(各个小时的均值最值) type为日子类型 周末、节假日等
#平时0 周末1 节假日2 无假期节假日3  不区分4  年前一周5  年中一周6  年后一周7
def get_dif_cate_features(df,cate):
    full_record = [116,46,17,3,182,7,7,7]  #每种类别理论上应该出现的次数 比如平时总共有116天
    theory_max = full_record[cate] #当前类型理论上的最大值

    day = get_day_info(df)  #获取一天总的人数的dataframe,下面几个同理
    morning = get_morning_info(df)
    afternoon = get_afternoon_info(df)
    evening = get_evening_info(df)
    night = get_night_info(df)

    day_happen_times = day.shape[0]  #获取当前类型下，实际多少天有记录，作为特征

    day_features = get_statistic_features(day,theory_max) #获取一天的统计特征,下面四个同理
    morning_features = get_statistic_features(morning,theory_max)
    afternoon_features = get_statistic_features(afternoon,theory_max)
    evening_features = get_statistic_features(evening,theory_max)
    night_features = get_statistic_features(night,theory_max)
    top_hours_features = get_top_hours(df) #返回最活跃的三个小时，按顺序是top1 top2 top3
    return day_features+morning_features+afternoon_features+evening_features+night_features+top_hours_features+[day_happen_times]


#获取所有训练集特征
def get_train_features():
    visit = pd.read_hdf('visit_hour_test_new.h5')
    visit_area = visit.groupby('area_id') #按照area_id进行分组
    length = len(visit_area)  #统计共有多少分组
    index = 0  #记录当前处理的是第多少个分组
    features = []
    for area_id,area in visit_area:
        print(length,index)  #显示总文件数和已处理文件数
        index += 1
        #获得该区域平时、周末、节假日、所有天的统计信息
        common = area[area['date_cate'] == 0]
        common_features = get_dif_cate_features(common, 0)
        weekend = area[area['date_cate'] == 1]
        weekend_features = get_dif_cate_features(weekend, 1)
        festival = area[area['date_cate'] == 2]
        festival_features = get_dif_cate_features(festival, 2)
        festival2 = area[area['date_cate'] == 3]
        festival2_features = get_dif_cate_features(festival2, 3)
        allday_features = get_dif_cate_features(area, 4) #这里不区分平时、节假日，全部在一起统计,类型计为4
        #获取该区域在年前、年中、年后的统计信息
        spring_before = area[(area['date']>=20190128) & (area['date']<=20190203)]
        spring_ing = area[(area['date']>=20190204) & (area['date']<=20190210)]
        spring_after = area[(area['date']>=20190211) & (area['date']<=20190217)]
        spring_before_features = get_dif_cate_features(spring_before, 5) #年前.年中.年后分别标记为5,6,7
        spring_ing_features = get_dif_cate_features(spring_ing, 6)
        spring_after_features = get_dif_cate_features(spring_after, 7)

        cur_features = common_features+weekend_features+festival_features+festival2_features+\
                       allday_features+spring_before_features + spring_ing_features + spring_after_features
        cur_features.append(area_id)
        features.append(cur_features)
    #column是所有的列名 0-7分别代表8种date_cate  平时、周末、节假日等，ABCDE分别代表全天、上午、下午、晚上、夜里, +全天活跃时、共计出现天数;
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

              'area_id']

    features = pd.DataFrame(features,columns = column)
    features.to_csv('test_feature.csv',index=False)


if __name__ == '__main__':
    #put_time_tag()   #这里需要要分两次操作否则会出现内存不足情况
    get_train_features()


