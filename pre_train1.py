import pandas as pd
import time
import os

#将当前txt文件中的访问记录整合为csv,后转换为list返回,精确到小时
def txt2csv_hour(filename):
    name = list(filename.strip('.txt').split('_'))
    area_id,func_id = list(map(int,name))
    file = os.path.join('./train',filename)
    result = []
    with open(file,'r') as f:
        for line in f:
            aa = line.strip().split('\t')[1]  #拿到某位用户的来访记录
            bb = aa.split(',')   #使用，获取每一天的来访记录
            for i in bb:
                cc = i.split('&')
                date = cc[0]
                hours = cc[1].split('|')  #获取一天中不同小时的来访记录
                hours = list(map(int,hours))
                for hour in hours:
                    year = int(date[:4])
                    month = int(date[4:6])
                    day = int(date[6:8])
                    result.append([year,month,day,hour])
        result = pd.DataFrame(result,columns = ['year','month','day','hour'])
        result2 = result.groupby(['year','month','day','hour'],as_index=False)['hour'].agg({'numbers':'count'})
        result2['area_id'] = [area_id] * result2.shape[0]
        result2['func_id'] = [func_id] * result2.shape[0]
        result2 = result2.values.tolist()
        return result2

#遍历文件夹内的所有txt文件，逐个变为list后，融入到大的list，最后转为visit.csv内
def get_visit_hour():
    visit = []
    file_lis = os.listdir('./train')
    cur_num = 0  #当前处理文件数
    start = time.time()
    file_lis1 = file_lis[:10000]
    file_lis2 = file_lis[10000:20000]
    file_lis3 = file_lis[20000:30000]
    file_lis4 = file_lis[30000:]
    file_num = len(file_lis4) #总文件数

    for file in file_lis4:
        cur_list = txt2csv_hour(file)
        #visit = pd.concat([visit,cur_df],sort=False,ignore_index=True)
        visit.extend(cur_list)
        print(file_num,cur_num)
        cur_num += 1
    visit = pd.DataFrame(visit,columns = ['year','month','day','hour','numbers','area_id','func_id'])
    visit.to_csv('visit_hour4.csv',index=False)
    end = time.time()
    print(end-start)




#将当前txt文件中的访问记录整合为csv,后转换为list返回,精确到天(统计人数，同一天一个人的多次记录算一次)
def txt2csv_day(filename):
    name = list(filename.strip('.txt').split('_'))
    area_id,func_id = list(map(int,name))
    file = os.path.join('./train',filename)
    result = []
    with open(file,'r') as f:
        for line in f:
            aa = line.strip().split('\t')[1]  #拿到某位用户的来访记录
            bb = aa.split(',')   #使用，获取每一天的来访记录
            for i in bb:
                cc = i.split('&')
                date = cc[0]
                year = int(date[:4])
                month = int(date[4:6])
                day = int(date[6:8])
                result.append([year,month,day])
        result = pd.DataFrame(result,columns = ['year','month','day'])
        result2 = result.groupby(['year','month','day'],as_index=False)['day'].agg({'numbers':'count'})
        result2['area_id'] = [area_id] * result2.shape[0]
        result2['func_id'] = [func_id] * result2.shape[0]
        result2 = result2.values.tolist()
        return result2

#遍历文件夹内的所有txt文件，逐个变为list后，融入到大的list，最后转为visit.csv(统计的是人次，一个人一天内多次访问记录算一次)
def get_visit_day():
    visit = []
    file_lis = os.listdir('./train')
    cur_num = 0  #当前处理文件数
    file_num = len(file_lis) #总文件数
    start = time.time()
    for file in file_lis:
        cur_list = txt2csv_day(file)
        #visit = pd.concat([visit,cur_df],sort=False,ignore_index=True)
        visit.extend(cur_list)
        print(file_num,cur_num)
        cur_num += 1

    visit = pd.DataFrame(visit,columns = ['year','month','day','numbers','area_id','func_id'])
    visit.to_csv('visit_day.csv',index=False)
    end = time.time()
    print(end-start)



#将当前txt文件中的访问记录整合为csv,后转换为list返回,精确到某个时段（一天内四个时段，同一时段内统计人次）
def txt2csv_quarter(filename):
    name = list(filename.strip('.txt').split('_'))
    area_id,func_id = list(map(int,name))
    file = os.path.join('./train',filename)
    result = []
    with open(file,'r') as f:
        for line in f:
            aa = line.strip().split('\t')[1]  #拿到某位用户的来访记录
            bb = aa.split(',')   #使用，获取每一天的来访记录
            for i in bb:
                cc = i.split('&')
                date = cc[0]
                year = int(date[:4])
                month = int(date[4:6])
                day = int(date[6:8])
                hours = cc[1].split('|')  #获取一天中不同小时的来访记录
                hours = set(map(int,hours))
                morning = {6,7,8,9,10,11}
                after = {12,13,14,15,16,17}
                evening = {18,19,20,21,22,23}
                night = {0,1,2,3,4,5}

                #某个时段内的多次访问算一次
                if hours & morning:
                    state = 1
                    result.append([year,month,day,state])
                if hours & after:
                    state = 2
                    result.append([year,month,day,state])
                if hours & evening:
                    state = 3
                    result.append([year,month,day,state])
                if hours & night:
                    state = 4
                    result.append([year,month,day,state])
        result = pd.DataFrame(result,columns = ['year','month','day','state'])
        result2 = result.groupby(['year','month','day','state'],as_index=False)['state'].agg({'numbers':'count'})
        result2['area_id'] = [area_id] * result2.shape[0]
        result2['func_id'] = [func_id] * result2.shape[0]
        result2 = result2.values.tolist()
        return result2

#遍历文件夹内的所有txt文件，逐个变为list后，融入到大的list，最后转为visit.csv,某个用户在某天的一个时段内的多次访问算一次
def get_visit_quarter():
    visit = []
    file_lis = os.listdir('./train')
    file_lis1 = file_lis[:20000]
    file_lis2 = file_lis[20000:]
    cur_num = 0  #当前处理文件数
    file_num = len(file_lis2) #总文件数
    start = time.time()
    for file in file_lis2:
        cur_list = txt2csv_quarter(file)
        #visit = pd.concat([visit,cur_df],sort=False,ignore_index=True)
        visit.extend(cur_list)
        print(file_num,cur_num)
        cur_num += 1

    visit = pd.DataFrame(visit,columns = ['year','month','day','state','numbers','area_id','func_id'])
    visit.to_csv('visit_quarter2.csv',index=False)
    end = time.time()
    print(end-start)


if __name__ == '__main__':
    #get_visit_hour() #精确到小时的人数统计，需要分批处理后合并
    #get_visit_day()  #精确到天的人数统计，这里统计的是人次，同一个人同一天内的多次访问算一次
    get_visit_quarter() #精确到时段（比如上午，下午），同一个人在一天内的某个时段多次访问算一次(分批处理)



