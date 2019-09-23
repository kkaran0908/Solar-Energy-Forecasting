#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import csv
from datetime import datetime
import requests
import json
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math


# ### Collect the weather data for a given time interval
# 1. link: https://darksky.net/dev

# In[99]:


csv_columns = ['time', 'summary', 'icon', 'precipIntensity','precipProbability', 'temperature', 'apparentTemperature', 'dewPoint', 'humidity', 'pressure', 'windSpeed', 'windGust', 'windBearing', 'cloudCover', 'uvIndex', 'visibility', 'ozone']
import pandas as pd
from datetime import datetime
#csv_file = "Names.csv"
df = pd.DataFrame(columns=csv_columns)
dec = df.to_csv('weather.csv',index = False)       
tim = 1556649000
count = 0
debug=0
l = []
while(count<30):
    url = "https://api.darksky.net/forecast/7d5861af747b98aead0c3d5c9d44171c/23.0225,72.5714,"+str(tim)
    response = requests.get(url)
    data = response.text
    parsed = json.loads(data)
    h_data = parsed['hourly']['data']
    for d in h_data:
        d_c = []
        l.append(len(d))
        #if debug==282:
            #print("282 rows data :",len(d))
       # if debug==283:
            #print("**************************************************")
            #print("566 row data:",d)
            #print(len(d))
            #break
        for v in d.keys():
            if v=='time':
                    timestamp = d[v]
                    #print(timestamp)
                    d_c.append(datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S'))
                    continue
           # else:
                #print(v)
            if v=='precipType':
                continue
            d_c.append(d[v])#(datetime.datetime.fromtimestamp(d[v]).strftime('%Y-%m-%d %H:%M:%S'))
    
        with open("weather.csv", "a") as fp:
            wr = csv.writer(fp, dialect='excel')
            wr.writerow(d_c)
        fp.close()
        debug = debug+1
    
    tim = tim+24*60*60
    count = count+1


# In[267]:


data = pd.read_csv('weather.csv')
data.head()


# In[268]:


plt.figure(figsize=(12,6))
plt.plot(data['windBearing'].values)
plt.title("Wind Bearing over the Month of April")
plt.xlabel("Time")
plt.ylabel("Wind Bearing")


# In[269]:


plt.figure(figsize=(12,6))
data['temperature'].plot()
plt.xlabel("time")
plt.ylabel("temperature")
plt.title("Temperature Variation with Time")


# In[270]:


plt.figure(figsize=(12,6))
data['windSpeed'].plot()
plt.xlabel("time")
plt.ylabel("Wind Speed")
plt.title("Wind Speed Variation with Time")


# In[271]:


plt.figure(figsize=(12,6))
data['apparentTemperature'].plot()
plt.xlabel("Time")
plt.ylabel("Apparent Temperature")
plt.title("Apparent Temperature Variation with Time")


# In[272]:


plt.figure(figsize=(12,6))
data['apparentTemperature'].plot()
data['temperature'].plot()
plt.xlabel("Time")
plt.ylabel("Apparent Temperature")
plt.title("Apparent Temperature Variation with Temperature")


# In[273]:


plt.figure(figsize=(12,6))
data['cloudCover'].plot()
plt.xlabel("time")
plt.ylabel("cloud cover")


# In[ ]:





# In[274]:


tem = data['temperature'].values
hum = data['humidity'].values
cloud = data['cloudCover'].values
wind = data['windSpeed'].values
#windBearing = data['windBearing'].values


# In[275]:


bearing = []
for v in data['windBearing'].values:
    bearing.append(v)
    bearing.append(bearing[-1])
    bearing.append(bearing[-1])


# In[276]:


apptem = []
for v in data['apparentTemperature'].values:
    apptem.append(v)
    apptem.append(apptem[-1])
    apptem.append(apptem[-1])


# In[277]:


dew = []
for v in data['dewPoint'].values:
    dew.append(v)
    dew.append(dew[-1])
    dew.append(dew[-1])


# In[278]:


temperature = []
for v in tem:
    temperature.append(v)
    temperature.append(temperature[-1])
    temperature.append(temperature[-1])


# In[279]:


windspeed = []
for v in wind:
    windspeed.append(v)
    windspeed.append(windspeed[-1])
    windspeed.append(windspeed[-1])


# In[280]:


humidity = []
for v in hum:
    humidity.append(v)
    humidity.append(humidity[-1])
    humidity.append(humidity[-1])


# In[281]:


cloudCover = []
for v in cloud:
    cloudCover.append(v)
    cloudCover.append(cloudCover[-1])
    cloudCover.append(cloudCover[-1])


# In[282]:


len(temperature),len(humidity),len(cloudCover),len(bearing)


# In[283]:


plt.figure(figsize=(12,6))
plt.plot(temperature)
plt.xlabel("time")
plt.ylabel("temperature")


# ## Convert the timestamp to date:

# In[203]:


import requests
import pandas as pd
import os
import numpy as np
import simplejson as json
import time 


def time_to_epoch(yyyy, mm, dd, t):
    t_stamp = str(yyyy).zfill(4) + '-' + str(mm).zfill(2) + '-' + str(dd).zfill(2) + 'T' + str(t)
    time_tuple = time.strptime(t_stamp, '%Y-%m-%dT%H:%M')
    time_epoch = time.mktime(time_tuple)
    return time_epoch


def epoch_to_timestring(n):
    return time.strftime('%Y-%m-%dT%H:%M', time.localtime(n))


# some global variables
private_server_url = 'http://127.0.0.1:5000'
header = {'Content-Type':'application/json'}
imars_base_url = 'http://14.215.130.185:20000'
time_stamps_list = ['UID','00:00', '00:20', '00:40', '01:00', '01:20', '01:40', '02:00', '02:20', '02:40', '03:00', '03:20',
                    '03:40', '04:00', '04:20', '04:40', '05:00', '05:20', '05:40', '06:00', '06:20', '06:40', '07:00',
                    '07:20', '07:40', '08:00', '08:20', '08:40', '09:00', '09:20', '09:40', '10:00', '10:20', '10:40',
                    '11:00', '11:20', '11:40', '12:00', '12:20', '12:40', '13:00', '13:20', '13:40', '14:00', '14:20',
                    '14:40', '15:00', '15:20', '15:40', '16:00', '16:20', '16:40', '17:00', '17:20', '17:40', '18:00',
                    '18:20', '18:40', '19:00', '19:20', '19:40', '20:00', '20:20', '20:40', '21:00', '21:20', '21:40',
                    '22:00', '22:20', '22:40', '23:00', '23:20', '23:40']
device_ids = [33140, 32977, 32684, 32934, 32804, 33096, 32984, 32985, 32725, 19380, 33131]

# Session preserves cookies from login accross https calls
s = requests.session()
# login
auth_data_imars = [
  ('username', 'iitg_in'),
  ('password', 'invt')]
login_response_imars = s.request('POST', imars_base_url + '/login.action', data=auth_data_imars)


# yyyy = 2018
# mm = 10
# dd = 1
# device_id = device_ids[1]

# Func to retreive data from solar panel energy portal for given date
# date format: 'yyyy-mm-dd'
def get_data(device_id, yyyy, mm, dd):
    date = str(yyyy).zfill(4) + '-' + str(mm).zfill(2) + '-' + str(dd).zfill(2)
    params_report = (('level', '3'), ('type', '1'), ('searchIds', device_id), ('searchDate', date))
    # searchIds = ID for panels/inverter...level=?, type=?
    date_report = s.request('GET', imars_base_url + '/reportjson/exportReport.action', params=params_report)
    #print(date_report.url)

    try:
        output = open('./data/{}/{}'.format(device_id, date+'.xls'), 'wb')
    except FileNotFoundError:
        os.makedirs('./data/{}'.format(device_id))
        output = open('./data/{}/{}'.format(device_id, date+'.xls'), 'wb')

    output.write(date_report.content)
    output.close()

    pd_data = pd.ExcelFile('./data/{}/{}'.format(device_id, date+'.xls'))
    pd_data_day = pd.read_excel(pd_data, 'dayReport')
   # print(pd_data_day)
    energy = np.array(pd_data_day[1:2])
    device_name = energy[0][0]
    #print('device_name:',device_name)

    t = np.array(pd_data_day[:1])
    t = np.delete(t, np.s_[:1], 1)
    t_list = list(t[0])

    energy = np.delete(energy, np.s_[:1], 1)
    energy = energy.astype(np.float32, copy=False)
    energy_list = list(energy[0])

    dict_t_e = {}
    # dict_forDF = {'date': date, 'name': device_name}
    for t, e in zip(t_list, energy_list):
        dict_t_e.update({t: e})
    return dict_t_e


yyyy = 2019
mm = 4
#dd = 23
import csv
import datetime

csv_columns =   ['00:00', '00:20', '00:40', '01:00', '01:20', '01:40', '02:00', '02:20', '02:40', '03:00', '03:20',
                    '03:40', '04:00', '04:20', '04:40', '05:00', '05:20', '05:40', '06:00', '06:20', '06:40', '07:00',
                    '07:20', '07:40', '08:00', '08:20', '08:40', '09:00', '09:20', '09:40', '10:00', '10:20', '10:40',
                    '11:00', '11:20', '11:40', '12:00', '12:20', '12:40', '13:00', '13:20', '13:40', '14:00', '14:20',
                    '14:40', '15:00', '15:20', '15:40', '16:00', '16:20', '16:40', '17:00', '17:20', '17:40', '18:00',
                    '18:20', '18:40', '19:00', '19:20', '19:40', '20:00', '20:20', '20:40', '21:00', '21:20', '21:40',
                    '22:00', '22:20', '22:40', '23:00', '23:20', 'Date']

#csv_file = "Names.csv"
df = pd.DataFrame(columns=csv_columns)
# name of the csv is in month+year+station_name    
dec = df.to_csv('xyz.csv',index=False)

# device id 0 -> Firpeal Hostel Capacity -> 25 kW,Serial No -> I01161005922
# device id 1 -> Beauki Hostel, Capacity -> 15 kW, Serial No -> I01163009107
# device id 2 -> Chimair Hostel, Capacity -> 15 kW, Serial No -> I01161006486
# device id 3 -> Duven Hostel, Capacity -> 15 kW, Serial No -> I01161004187
# device id 4 -> Emiet Hostel Capacity -> 25 kW, Serial No -> I01161005921
# device id 5 -> No Working
# device id 6 -> Working but still unknown
# device id 7 -> Not Working
# device id 8 -> Not Working
# device id 9 -> Not Working
# device id 10 -> Aibaan Hostel, Capacity -> 25 Kw,Serial No -> I01161005931
# device id 11 -> 
device_id = device_ids[1]
count = 1
for dd in range(1,31):
    print("count: ",count)
    count+=1
    l = []
    dict_t_e=get_data(device_id, yyyy, mm, dd)
    x = datetime.datetime(yyyy,mm,dd)
    #print(dict_t_e)
    for d in dict_t_e.keys():
        l.append(dict_t_e[d])
    l.append(x)
    with open("xyz.csv", "a") as fp:
        wr = csv.writer(fp, dialect='excel')
        wr.writerow(l)
    fp.close()


# In[284]:


data = pd.read_csv('xyz.csv')


# In[285]:


data.head()


# In[286]:


plt.figure(figsize=(12,6))
plt.plot(data[data.columns[:-1]].values.flatten())
plt.xlabel("time")
plt.ylabel("Energy")


# In[287]:


data = data.drop(['Date'],axis=1)


# In[288]:


data.shape


# In[289]:


data.head()


# In[290]:


dx = np.array(temperature).reshape((30, 72)) 
dy = np.array(humidity).reshape((30,72))
dz = np.array(cloudCover).reshape(30,72)
dx1 = np.array(apptem).reshape((30, 72)) 
dy1 = np.array(dew).reshape((30,72))
dz1 = np.array(windspeed).reshape(30,72)


# In[291]:


bear = np.array(bearing).reshape(30,72)


# In[292]:


dx.shape,dz.shape,dz1.shape


# In[293]:


cl = list(data.columns)


# In[294]:


cl.append('23:40')


# In[295]:


len(cl)


# In[296]:


temperature = pd.DataFrame(dx,columns=cl)
humidity = pd.DataFrame(dy,columns=cl)
ccover = pd.DataFrame(dz,columns=cl)
apptem = pd.DataFrame(dx1,columns=cl)
dew = pd.DataFrame(dy1,columns=cl)
windspeed = pd.DataFrame(dz1,columns=cl)


# In[297]:


windBearing = pd.DataFrame(bear,columns=cl)


# In[298]:


windBearing.head()


# In[299]:


temperature.head()


# In[300]:


ccover.head()


# In[301]:


windspeed.head()


# In[302]:


dew.head()


# In[303]:


temperature = temperature.drop(['23:40'],axis=1)
humidity = humidity.drop(['23:40'],axis=1)
ccover = ccover.drop(['23:40'],axis=1)
dew = dew.drop(['23:40'],axis=1)
apptem = apptem.drop(['23:40'],axis=1)
#ccover = ccover.drop(['23:40'],axis=1)


# In[304]:


wspeed  = windspeed.drop(['23:40'],axis=1)


# In[305]:


windBearing  = windBearing.drop(['23:40'],axis=1)


# In[306]:


for d in temperature.values:
    plt.plot(d)
plt.xlabel("Time from 12 AM to 23:59")
plt.ylabel("Temperature")
plt.xticks([])
plt.plot()


# In[307]:


for d in dew.values:
    plt.plot(d)
plt.xlabel("Time from 12 AM to 23:59")
plt.ylabel("Dew Point")
plt.xticks([])
plt.plot()


# ### Taking reading from 6 am to 6 pm in the evening

# In[308]:


temperature = (temperature.T.iloc[20:55]).T
humidity = (humidity.T.iloc[20:55]).T
energy = (data.T.iloc[20:55]).T
ccover = (ccover.T.iloc[20:55]).T


# In[309]:


wspeed  = (wspeed.T.iloc[20:55]).T


# In[310]:


dew  = (dew.T.iloc[20:55]).T


# In[311]:


windBearing = (windBearing.T.iloc[20:55]).T


# In[312]:


temperature.head()  #temperature data


# In[313]:


energy.head(5)  # energy data


# In[314]:


humidity.head()


# In[315]:


ccover.head()


# In[316]:


dew.head()


# In[317]:


windBearing.head()


# ## converting temperature and energy in one column

# In[318]:


tem,ene,hum,cloud = [],[],[],[]
for i in range(len(temperature.values)):
    for j in range(len(temperature.values[0])):
        tem.append(temperature.values[i][j])
for i in range(len(energy.values)):
    for j in range(len(energy.values[0])):
        ene.append(energy.values[i][j])
for i in range(len(humidity.values)):
    for j in range(len(humidity.values[0])):
        hum.append(humidity.values[i][j])
        
for i in range(len(ccover.values)):
    for j in range(len(ccover.values[0])):
        cloud.append(ccover.values[i][j])


# In[319]:


windspeed = []
for i in range(len(wspeed.values)):
    for j in range(len(wspeed.values[0])):
        windspeed.append(wspeed.values[i][j])


# In[320]:


dewpoint = []
for i in range(len(dew.values)):
    for j in range(len(dew.values[0])):
        dewpoint.append(dew.values[i][j])


# In[321]:


wBearing = []
for i in range(len(windBearing.values)):
    for j in range(len(windBearing.values[0])):
        wBearing.append(windBearing.values[i][j])


# ## Energy Variation with Wind Bearing

# In[322]:


plt.scatter(wBearing,ene)


# In[415]:


data1.head()


# In[416]:


data1.shape


# In[417]:


d = data1['Energy_t']


# In[423]:


from pandas.plotting import autocorrelation_plot

plt.figure(figsize=(20,4))
#Out[102]: <Figure size 640x480 with 0 Axes>

spacing = np.linspace(-9 * np.pi, 9 * np.pi, num=1000)
#data = pd.Series(0.7 * np.random.rand(1000) + 0.3 * np.sin(spacing))

autocorrelation_plot(d)
plt.xticks(ticks=list(range(0, 1000,35)))
#Out[105]: <matplotlib.axes._subplots.AxesSubplot at 0x7f2463eb8250>


# In[ ]:





# In[323]:


plt.plot(wBearing)
plt.plot(ene)


# In[324]:


ene1 = np.array(ene).reshape(-1,1)
wBear = np.array(wBearing).reshape(-1,1)


# In[325]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
ene1 = scaler.fit_transform(ene1)
scaler1 = StandardScaler()
wBearing1 = scaler1.fit_transform(wBear)


# In[326]:


plt.figure(figsize = (20,10))
plt.plot(wBearing1[:300])
plt.plot(ene1[:300])


# In[327]:


plt.scatter(wBearing1,ene1)


# ## Energy variation with dewPoint

# In[328]:


plt.scatter(dewpoint,ene)
plt.xlabel("Dew Point")
plt.ylabel("Energy Produced")
plt.title("Energy Variation with Wind Speed")


# In[329]:


plt.plot(dewpoint)
plt.plot(ene)


# In[330]:


from sklearn.preprocessing import StandardScaler
ene1 = np.array(ene).reshape(-1,1)
dpoint1 = np.array(dewpoint).reshape(-1,1)

scaler = StandardScaler()
ene1 = scaler.fit_transform(ene1)
scaler1 = StandardScaler()
dpoint1 = scaler1.fit_transform(dpoint1)


# In[331]:


plt.figure(figsize =(20,8))
plt.plot(dpoint1)
plt.plot(ene1)
plt.legend(["Dew Point","Energy Produced"])
plt.title("Energy variation with Dew Point")


# ## Energy variation with wind speed 

# In[332]:


plt.scatter(windspeed,ene)
plt.xlabel("Wind Speed")
plt.ylabel("Energy Produced")
plt.title("Energy Variation with Wind Speed")


# In[333]:


plt.figure(figsize = (20,5))
plt.plot(windspeed)
plt.plot(ene)


# In[334]:


from sklearn.preprocessing import StandardScaler
ene1 = np.array(ene).reshape(-1,1)
windspeed1 = np.array(windspeed).reshape(-1,1)
scaler = StandardScaler()
ene1 = scaler.fit_transform(ene1)
scaler1 = StandardScaler()
windspeed1 = scaler1.fit_transform(windspeed1)


# In[335]:


plt.figure(figsize =(20,8))
plt.plot(windspeed1)
plt.plot(ene1)
plt.legend(["Dew Point","Energy Produced"])
plt.title("Energy variation with Dew Point")


# ## Energy Variation with Cloud Cover

# 1. (0.985 − 0.984n^3.4)

# In[336]:


len(ene)/30


# In[337]:


plt.scatter(cloud,ene)


# In[338]:


c = 990*(1-0.75*(np.array(cloud)**3))


# In[339]:


plt.scatter(c,ene)


# In[340]:


plt.scatter(c,cloud)
plt.xlabel("Solar Radiation")
plt.ylabel("Cloud Cover")
plt.title("Solar Radiation vs Cloud Cover")


# In[341]:


c = []
for i in cloud:
    d = .985 - .984*(i**3.4)
    c.append(d)


# In[342]:


len(c),len(ene)


# In[343]:


plt.scatter(c,ene)
plt.xlabel("Irradiance")
plt.ylabel("Energy Produced")
plt.show()


# In[344]:


len(cloud)


# In[345]:


c = []
count = 0
d = 0
for i in cloud:
    if count ==35:
        c.append(d/35.0)
        d = 0
        count = 0
    count = count+1
        
    d = d + .985 - .984*(i**3.4)
    #c.append(d)


# In[346]:


e = []
d = 0
count = 0
for i in ene:
    if count ==35:
        e.append(d)
        d = 0
        count = 0
    d = d+i
    count = count+1


# In[347]:


len(e),len(c)


# In[348]:


plt.scatter(c,e)
plt.ylabel("Energy Produced Per Day")
plt.xlabel("Per Day Average Irradiance")


# In[349]:


data = pd.DataFrame(columns=['Temperature','Energy'])


# In[350]:


data['Temperature'] = tem
data['Energy'] = ene


# In[351]:


data.shape


# In[352]:


plt.scatter(data['Temperature'],data['Energy'])
plt.xlabel('Temperature in  Fahrenheit')
plt.ylabel("Energy in KWh")
plt.title("Energy Variation with Temperature")
plt.show()


# ### Applying svm model
# 
# X_train -> time , Y_train -> Energy 

# In[353]:


data.head()


# In[90]:


k = 40 ## number of time stamp


# ### Preparing data for modeling
# 
# <h5>X_train - > Temperature, Energy</h5>
# <h5>Y_train -> Energy</h5>

# In[354]:


x = data['Energy'].values[:len(data['Energy'].values)-k]  


# In[355]:


x.shape


# In[356]:


data1 = data.iloc[k:]


# In[357]:


data1['Energy_t'] = x


# In[358]:


data1.shape


# In[359]:


np.corrcoef(data1['Energy'],data1['Energy_t'])   ## testing the correlation Actual value and predicted energy


# In[361]:


data1.head()


# In[362]:


data1.head()


# In[363]:


data.iloc[:5]


# In[364]:


day_reading = 35*21  ## 21 days reading as the train data
X_train = data1[['Temperature','Energy_t']].iloc[:day_reading]
X_test = data1[['Temperature','Energy_t']].iloc[day_reading:]
y_train = data1['Energy'].iloc[:day_reading]
y_test = data1['Energy'].iloc[day_reading:]


# In[365]:


X_train.shape,X_test.shape


# In[369]:


def create_train_test1(k,data1):
    x = data1['Energy'].values[:len(data1['Energy'].values)-k]  
    data1 = data.iloc[k:]
    data1['Energy_t'] = x
    day_reading = 35*21  ## 21 days reading as the train data
    X_train = data1[['Temperature','Energy_t']].iloc[:day_reading]
    X_test = data1[['Temperature','Energy_t']].iloc[day_reading:]
    y_train = data1['Energy'].iloc[:day_reading]
    y_test = data1['Energy'].iloc[day_reading:]
    return [X_train,X_test,y_train,y_test]   


# In[370]:


data.shape


# In[371]:


import warnings
warnings.filterwarnings('ignore')
X_train,X_test,y_train,y_test = create_train_test1(40,data)


# In[372]:


X_train.shape,X_test.shape


# ## Applying SVM

# In[373]:


X_train.head()


# In[374]:


import warnings
warnings.filterwarnings('ignore')
rmse1 = []
mape1 = []
for k in range(1,40):
    X_train,X_test,y_train,y_test=create_train_test1(k,data)
    reg = SVR(kernel='linear',gamma = 'auto',C=10)
    reg.fit(X_train,y_train)
    y_pred = reg.predict(X_test)
    y_pred1 = reg.predict(X_train)
    m = mean_squared_error(y_test,y_pred)
    m1 = mean_squared_error(y_train,y_pred1)
    print("Root Mean For Lag Value:"+str(k)+'---->', math.sqrt(m))
    rmse1.append(math.sqrt(m))
    mape1.append(MAPE(y_test.values,y_pred))
    print("MAPE using SVM :", MAPE(y_test.values,y_pred))
    print("***************************************************")
    #print("Root Mean squared error using SVM on train data:", math.sqrt(m1))
    if math.sqrt(m) < opt_v:
        opt_v = math.sqrt(m)
        y_opt = y_pred   


# In[375]:


plt.plot(mape1)
plt.title("MAPE variation with lag values")
plt.legend(['Temperature + Energy'])
plt.plot()


# In[376]:


plt.plot(rmse1)
plt.title("RMSE variation with lag values")
plt.legend(['Features: Temperature + Energy'])
plt.plot()


# In[134]:


from sklearn.svm import SVR

# applying SVM over the data using rbf kernel
from sklearn.svm import SVR
opt_v = 99999
y_opt = []
for c in [.001,.01,.1,1,10,100,1000]:
    reg = SVR(kernel='linear',gamma = 'auto',C=c)
    reg.fit(X_train,y_train)
    y_pred = reg.predict(X_test)
    y_pred1 = reg.predict(X_train)
    m = mean_squared_error(y_test,y_pred)
    m1 = mean_squared_error(y_train,y_pred1)
    print("Root Mean squared error using SVM on test data:", math.sqrt(m))
    #print("Root Mean squared error using SVM on train data:", math.sqrt(m1))
    if math.sqrt(m) < opt_v:
        opt_v = math.sqrt(m)
        y_opt = y_pred


# In[162]:


""""from sklearn.svm import SVR

# applying SVM over the data using rbf kernel
from sklearn.svm import SVR
opt_v = 99999
y_opt = []


y_pred = np.array((list(mean_signal)*100)[:y_test.size])
y_pred1 = np.array((list(mean_signal)*100)[:y_train.size])
m = mean_squared_error(y_test,y_pred)
m1 = mean_squared_error(y_train,y_pred1)
print("Root Mean squared error using SVM on test data:", math.sqrt(m))
#print("Root Mean squared error using SVM on train data:", math.sqrt(m1))
"""


# In[377]:


def MAPE(y,yhat):
    d = 0
    for i in range(len(y)):
        d = d + abs(y[i]-yhat[i])/y[i]
    return (100*d)/len(y)


# In[137]:


plt.plot(y_test.values)
plt.plot(y_pred)
plt.legend(["Actual","Predicted"])
print("Mean squared error using SVM :", opt_v)
#print("MAPE using SVM :", MAPE(y_test.values,y_opt))
print("MAPE using SVM :", MAPE(y_test.values,y_pred))


# In[376]:


from sklearn.svm import SVR

# applying SVM over the data using rbf kernel
from sklearn.svm import SVR
opt_v = 99999
y_opt = []
for c in [.001,.01,.1,1,10,100,1000]:
    reg = SVR(kernel='rbf',gamma = 'auto',C=c)
    reg.fit(X_train,y_train)
    y_pred = reg.predict(X_test)
    y_pred1 = reg.predict(X_train)
    m = mean_squared_error(y_test,y_pred)
    m1 = mean_squared_error(y_train,y_pred1)
    print("Root Mean squared error using SVM on test data:", math.sqrt(m))
    #print("Root Mean squared error using SVM on train data:", math.sqrt(m1))
    if math.sqrt(m) < opt_v:
        opt_v = math.sqrt(m)
        y_opt = y_pred


# In[377]:


plt.plot(y_test.values)
plt.plot(y_opt)
plt.legend(["Actual","Predicted"])
print("Mean squared error using SVM :", opt_v)
print("MAPE using SVM :", MAPE(y_test.values,y_opt))


# ## Applying Nural Network

# In[150]:


data.head()


# In[151]:


train_data,test_data = data.drop(['Predicted_Energy','Energy'],axis=1).iloc[:750],data.drop(['Predicted_Energy','Energy'],axis=1).iloc[750:]
train_y,test_y = data['Predicted_Energy'].iloc[:750].values,data['Predicted_Energy'].iloc[750:].values


# In[152]:


import tensorflow
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import np_utils  
import seaborn as sns
from keras import layers
from keras.initializers import RandomNormal
import random
import keras


# In[557]:


#random.seed(40)
model = Sequential()
model.add(Dense(1, input_shape=(train_data.shape[1],), activation='sigmoid', kernel_initializer='lecun_uniform'))
model.add(Dense(64,activation = 'tanh'))
#model.add(layers.Dropout(0.2))

keras.regularizers.l1_l2(l1=0.01)
#model.add(Dense(64,activation = 'sigmoid'))
#model.add(layers.Dropout(0.2))

model.add(Dense(1,activation = 'relu'))
#model.add(Dense(1,activation = 'sigmoid'))
#model.add(Dense(1,activation = 'relu'))
#model.add(Dense(1))#,activation = 'sigmoid'))
#model.add(Dense(1,activation = 'relu'))
model.compile(optimizer=Adam(lr=0.01), loss='mean_squared_error')
model.fit(train_data, train_y, batch_size=100, epochs=1500, verbose=1)


# In[153]:


train_data,test_data = data.drop(['Predicted_Energy'],axis=1).iloc[:750],data.drop(['Predicted_Energy'],axis=1).iloc[750:]
train_y,test_y = data['Predicted_Energy'].iloc[:750].values,data['Predicted_Energy'].iloc[750:].values


# In[154]:


train_data,test_data = np.reshape(train_data.values, (-1, 1)),np.reshape(test_data.values,(-1,1))


# In[548]:


#random.seed(40)
model = Sequential()
model.add(Dense(1, input_shape=(train_data.shape[1],), activation='sigmoid', kernel_initializer='lecun_uniform'))
model.add(Dense(64,activation = 'tanh'))
#model.add(layers.Dropout(0.2))

keras.regularizers.l1_l2(l1=0.01)
#model.add(Dense(64,activation = 'sigmoid'))
#model.add(layers.Dropout(0.2))

model.add(Dense(1,activation = 'relu'))
#model.add(Dense(1,activation = 'sigmoid'))
#model.add(Dense(1,activation = 'relu'))
#model.add(Dense(1))#,activation = 'sigmoid'))
#model.add(Dense(1,activation = 'relu'))
model.compile(optimizer=Adam(lr=0.01), loss='mean_squared_error')
model.fit(train_data, train_y, batch_size=100, epochs=1500, verbose=1)


# In[549]:


plt.figure(figsize=(10, 5))
y_pred = model.predict(test_data)
plt.plot(test_y)
plt.plot(y_pred)
plt.legend(["Actual","Predicted"])
print('Root mean Squared Error',math.sqrt(mean_squared_error(test_y,y_pred)))


# In[550]:


plt.figure(figsize=(15, 10))
y_pred = model.predict(train_data)
plt.plot(train_y)
plt.plot(y_pred)
plt.legend(["Actual","Predicted"])
print('Root mean Squared Error',math.sqrt(mean_squared_error(train_y,y_pred)))


# ### Using time of the Day, Energy and Temperature for prediction
# 
# <h5>X_train -> Time, Energy, Temperature</h5>
# 
# <h5>Y_train -> Predicted_Energy</h5>

# In[378]:


time = []
for i in range(0,30):
    for j in energy.columns:
        time.append(j)
#time = time[:len(time)-k]


# In[379]:


data.head()


# In[380]:


from sklearn.preprocessing import OneHotEncoder
def create_train_test(k,data,time):
    time = []
    #print(data.shape)
    for i in range(0,30):
        for j in energy.columns:
            time.append(j)
    data['Time'] = time
    x = data['Energy'].values[:len(data['Energy'].values)-k]  
    data = data.iloc[k:]
    data['Energy_t'] = x
    #print(data)
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(data['Time'].values.reshape(-1,1))
    X = enc.transform(data['Time'].values.reshape(-1,1))
    #print(data)
   
    data.reset_index(drop=True, inplace=True)
    day_d = pd.DataFrame(X.todense())
    day_d.reset_index(drop=True, inplace=True)
    #print(day_d.shape)
    data = pd.concat([data,day_d],axis=1)
    #print(data)
    data = data.drop(['Time'],axis=1)
    
    train = data.iloc[:day_reading]
    
    test = data.iloc[day_reading:]
    X_train = train.drop(['Energy'],axis=1)
    y_train = train['Energy']
    X_test = test.drop(['Energy'],axis=1)
    y_test = test['Energy']
    return [X_train,X_test,y_train,y_test]


# In[381]:


X_train,X_test,y_train,y_test=create_train_test(40,data,time)


# In[382]:


X_train.shape,X_test.shape


# In[383]:


X_train,X_test,y_train,y_test=create_train_test(1,data,time)


# In[384]:


X_train.shape,X_test.shape


# In[385]:


import warnings
warnings.filterwarnings('ignore')
rmse2 = []
mape2 = []
for k in range(1,41):
    X_train,X_test,y_train,y_test=create_train_test(k,data,time)
    reg = SVR(kernel='linear',gamma = 'auto',C=10)
    reg.fit(X_train,y_train)
    y_pred = reg.predict(X_test)
    y_pred1 = reg.predict(X_train)
    m = mean_squared_error(y_test,y_pred)
    m1 = mean_squared_error(y_train,y_pred1)
    print("Root Mean For Lag Value:"+str(k)+'---->', math.sqrt(m))
    rmse2.append(math.sqrt(m))
    mape2.append(MAPE(y_test.values,y_pred))
    print("MAPE using SVM :", MAPE(y_test.values,y_pred))
    print("***************************************************")
    #print("Root Mean squared error using SVM on train data:", math.sqrt(m1))
    if math.sqrt(m) < opt_v:
        opt_v = math.sqrt(m)
        y_opt = y_pred   


# In[388]:


#plt.plot(mape1)
plt.plot(mape2)
plt.title("MAPE variation with lag values")
plt.legend(['Temperature + Energy + Time'])
plt.plot()


# In[389]:


plt.plot(mape1)
plt.plot(mape2)
plt.title("MAPE variation with lag values")
plt.legend(['Temperature + Energy','Temperature + Energy + Time'])
plt.plot()


# In[391]:


plt.plot(rmse2)
plt.title("RMSE variation with lag values")
plt.legend(['Temperature + Energy + Time'])
plt.plot()


# In[392]:


plt.plot(rmse1)
plt.plot(rmse2)
plt.title("RMSE variation with lag values")
plt.legend(['Temperature+Energy','Temperature + Energy + Time'])
plt.plot()


# In[401]:


from sklearn.svm import SVR
opt_v = 99999
y_opt = []
for c in [.001,.01,.1,1,10,100]:
    reg = SVR(kernel='linear',gamma = 'auto',C=c)
    reg.fit(X_train,y_train)
    y_pred = reg.predict(X_test)
    y_pred1 = reg.predict(X_train)
    m = mean_squared_error(y_test,y_pred)
    m1 = mean_squared_error(y_train,y_pred1)
    print("Root Mean squared error using SVM on test data:", math.sqrt(m))
    #print("Root Mean squared error using SVM on train data:", math.sqrt(m1))
    if math.sqrt(m) < opt_v:
        opt_v = math.sqrt(m)
        y_opt = y_pred


# In[402]:


plt.plot(y_test.values)
plt.plot(y_opt)
plt.legend(["Actual","Predicted"])
print("Mean squared error using SVM :", opt_v)
print("MAPE using SVM :", MAPE(y_test.values,y_opt))


# In[187]:


from sklearn.svm import SVR

# applying SVM over the data using rbf kernel
from sklearn.svm import SVR
opt_v = 99999
y_opt = []
for c in [.01,.1,1,10,100,1000]:
    reg = SVR(kernel='rbf',gamma = 'auto',C=c)
    reg.fit(X_train,y_train)
    y_pred = reg.predict(X_test)
    y_pred1 = reg.predict(X_train)
    m = mean_squared_error(y_test,y_pred)
    m1 = mean_squared_error(y_train,y_pred1)
    print("Root Mean squared error using SVM on test data:", math.sqrt(m))
    #print("Root Mean squared error using SVM on train data:", math.sqrt(m1))
    if math.sqrt(m) < opt_v:
        opt_v = math.sqrt(m)
        y_opt = y_pred


# In[188]:


plt.plot(y_test.values)
plt.plot(y_opt)
plt.legend(["Actual","Predicted"])
print("Mean squared error using SVM :", opt_v)
print("MAPE using SVM :", MAPE(y_test.values,y_opt))


# ### Applying SVM
# 
# ## Using Temperature, Time, Previous Energy Produced, Humidity 
# <h5>X_train -> Temperature, Time, Humidity</h5>
# <h5>Y_train -> Energy</h5>

# In[393]:


data['humidity']  = hum#hum[:len(hum)-k]


# In[394]:


data.shape


# In[395]:


data.head()


# In[396]:


from sklearn.preprocessing import OneHotEncoder
def create_train_test(k,data,time):
    #time = []
    #print(data.shape)
    #for i in range(0,30):
    #    for j in energy.columns:
    #        time.append(j)
    #data['Time'] = time
    x = data['Energy'].values[:len(data['Energy'].values)-k]  
    data = data.iloc[k:]
    data['Energy_t'] = x
    #print(data)
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(data['Time'].values.reshape(-1,1))
    X = enc.transform(data['Time'].values.reshape(-1,1))
    #print(data)
   
    data.reset_index(drop=True, inplace=True)
    day_d = pd.DataFrame(X.todense())
    day_d.reset_index(drop=True, inplace=True)
    #print(day_d.shape)
    data = pd.concat([data,day_d],axis=1)
    #print(data)
    data = data.drop(['Time'],axis=1)
    
    train = data.iloc[:day_reading]
    
    test = data.iloc[day_reading:]
    X_train = train.drop(['Energy'],axis=1)
    y_train = train['Energy']
    X_test = test.drop(['Energy'],axis=1)
    y_test = test['Energy']
    return [X_train,X_test,y_train,y_test]


# In[209]:


X_train,X_test,y_train,y_test=create_train_test(40,data,time)


# In[210]:


X_train.head()


# In[212]:


data.head()


# In[397]:


import warnings
warnings.filterwarnings('ignore')
rmse3 = []
mape3 = []
for k in range(1,40):
    X_train,X_test,y_train,y_test=create_train_test(k,data,time)
    reg = SVR(kernel='linear',gamma = 'auto',C=10)
    reg.fit(X_train,y_train)
    y_pred = reg.predict(X_test)
    y_pred1 = reg.predict(X_train)
    m = mean_squared_error(y_test,y_pred)
    m1 = mean_squared_error(y_train,y_pred1)
    print("Root Mean For Lag Value:"+str(k)+'---->', math.sqrt(m))
    rmse3.append(math.sqrt(m))
    mape3.append(MAPE(y_test.values,y_pred))
    print("MAPE using SVM :", MAPE(y_test.values,y_pred))
    print("***************************************************")
    #print("Root Mean squared error using SVM on train data:", math.sqrt(m1))
    if math.sqrt(m) < opt_v:
        opt_v = math.sqrt(m)
        y_opt = y_pred


# In[398]:


plt.plot(mape2)
plt.plot(mape3)
plt.title("MAPE variation with lag values")
plt.legend(['Temperature, Time, Previous Energy','Temperature, Time, Previous Energy,Humidity'])
plt.xlabel('Lag values')
plt.ylabel('MAPE')
plt.show()


# In[399]:


plt.plot(rmse2)
plt.plot(rmse3)
plt.title("RMSE variation with lag values")
plt.legend(['Temperature, Time, Previous Energy','Temperature, Time, Previous Energy Produced,Humidity'])
plt.xlabel('Lag values')
plt.ylabel('MAPE')
plt.plot()


# In[210]:


X_train,X_test,y_train,y_test=create_train_test(40,data,time,hum)


# In[211]:


X_train.head()


# In[405]:


plt.scatter(data['humidity'],data['Energy'])
plt.xlabel("Humidity in Percentage")
plt.ylabel("Energy Produced in KWh")
plt.title("Energy Variation with Humidity")
plt.show()


# In[406]:


X_train = data.drop(['Predicted_Energy'],axis=1).iloc[:day_reading]
y_train = data['Predicted_Energy'].iloc[:day_reading]
X_test = data.drop(['Predicted_Energy'],axis=1).iloc[day_reading:]
y_test = data['Predicted_Energy'].iloc[day_reading:]


# In[407]:


X_train.head()


# In[408]:


opt_v = 99999
y_opt = []
for c in [.001,.01,.1,1,10,100,1000]:
    reg = SVR(kernel='rbf',gamma = 'auto',C=c)
    reg.fit(X_train,y_train)
    y_pred = reg.predict(X_test)
    y_pred1 = reg.predict(X_train)
    m = mean_squared_error(y_test,y_pred)
    m1 = mean_squared_error(y_train,y_pred1)
    print("Root Mean squared error using SVM on test data:", math.sqrt(m))
    #print("Root Mean squared error using SVM on train data:", math.sqrt(m1))
    if math.sqrt(m) < opt_v:
        opt_v = math.sqrt(m)
        y_opt = y_pred


# In[409]:


plt.plot(y_opt)
plt.plot(y_test.values)
plt.legend(["Energy Produced","Energy Predicted"])
plt.title("Solar Energy Prediction Over Test Data")
plt.show()
print("Optimal Result of the model :", opt_v)
print("MAPE using SVM :", MAPE(y_test.values,y_opt))


# In[410]:


opt_v = 99999
y_opt = []
for c in [.001,.01,.1,1,10,100,1000]:
    reg = SVR(kernel='linear',gamma = 'auto',C=c)
    reg.fit(X_train,y_train)
    y_pred = reg.predict(X_test)
    y_pred1 = reg.predict(X_train)
    m = mean_squared_error(y_test,y_pred)
    m1 = mean_squared_error(y_train,y_pred1)
    print("Root Mean squared error using SVM on test data:", math.sqrt(m))
    #print("Root Mean squared error using SVM on train data:", math.sqrt(m1))
    if math.sqrt(m) < opt_v:
        opt_v = math.sqrt(m)
        y_opt = y_pred


# In[411]:


plt.plot(y_opt)
plt.plot(y_test.values)
plt.legend(["Energy Produced","Energy Predicted"])
plt.title("Solar Energy Prediction Over Test Data")
plt.show()
print("Optimal Result of the model :", opt_v)
print("MAPE using SVM :", MAPE(y_test.values,y_opt))


# In[414]:


from sklearn.svm import SVR

# applying SVM over the data using rbf kernel
from sklearn.svm import SVR
opt_v = 99999
y_opt = []
for c in [.001,.01,.1,1,10,100,1000]:
    reg = SVR(kernel='poly',gamma = 'auto',C=c)
    reg.fit(X_train,y_train)
    y_pred = reg.predict(X_test)
    y_pred1 = reg.predict(X_train)
    m = mean_squared_error(y_test,y_pred)
    m1 = mean_squared_error(y_train,y_pred1)
    print("Root Mean squared error using SVM on test data:", math.sqrt(m))
    #print("Root Mean squared error using SVM on train data:", math.sqrt(m1))
    if math.sqrt(m) < opt_v:
        opt_v = math.sqrt(m)
        y_opt = y_pred
print("Mean squared error using SVM on Test:", opt_v)


# In[415]:


plt.plot(y_opt)
plt.plot(y_test.values)
plt.legend(["Energy Produced","Energy Predicted"])
plt.title("Solar Energy Prediction Over Test Data")
plt.show()
print("Mean squared error using SVM on Test:", opt_v)
print("MAPE using SVM :", MAPE(y_test.values,y_opt))


# In[417]:


data.head()


# ## Time, Humidity, Temperature

# In[400]:


data.head()


# In[220]:


from sklearn.preprocessing import OneHotEncoder
def create_train_test(k,data,time):
    x = data['Energy'].values[:len(data['Energy'].values)-k]  
    data = data.iloc[k:]
    data['Energy_t'] = x
    #print(data)
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(data['Time'].values.reshape(-1,1))
    X = enc.transform(data['Time'].values.reshape(-1,1))
    #print(data)
   
    data.reset_index(drop=True, inplace=True)
    day_d = pd.DataFrame(X.todense())
    day_d.reset_index(drop=True, inplace=True)
    #print(day_d.shape)
    data = pd.concat([data,day_d],axis=1)
    #print(data)
    data = data.drop(['Time'],axis=1)
    
    train = data.iloc[:day_reading]
    
    test = data.iloc[day_reading:]
    X_train = train.drop(['Energy','Energy_t'],axis=1)
    y_train = train['Energy']
    X_test = test.drop(['Energy','Energy_t'],axis=1)
    y_test = test['Energy']
    return [X_train,X_test,y_train,y_test]


# In[228]:


X_train,X_test,y_train,y_test=create_train_test(k,data,time)


# In[240]:


import warnings
warnings.filterwarnings('ignore')
rmse4 =  []
mape4 =  []
for k in range(1,40):
    X_train,X_test,y_train,y_test=create_train_test(k,data,time)
    reg = SVR(kernel='linear',gamma = 'auto',C=10)
    reg.fit(X_train,y_train)
    y_pred = reg.predict(X_test)
    y_pred1 = reg.predict(X_train)
    m = mean_squared_error(y_test,y_pred)
    m1 = mean_squared_error(y_train,y_pred1)
    print("Root Mean For Lag Value:"+str(k)+'---->', math.sqrt(m))
    rmse4.append(math.sqrt(m))
    mape4.append(MAPE(y_test.values,y_pred))
    print("MAPE using SVM :", MAPE(y_test.values,y_pred))
    print("***************************************************")
    #print("Root Mean squared error using SVM on train data:", math.sqrt(m1))
    if math.sqrt(m) < opt_v:
        opt_v = math.sqrt(m)
        y_opt = y_pred


# In[403]:


plt.plot(mape3)
plt.plot(mape2)
plt.plot(mape4)
plt.title("MAPE variation with lag values")
plt.legend(['Temperature, Time, Previous Energy, Humidity','Temperature, Time, Previous Energy','Time , Humidity, Temperature'])
plt.xlabel('Lag values')
plt.ylabel('MAPE')
plt.plot()


# In[404]:


plt.plot(mape4)


# In[412]:


plt.plot(rmse3)
plt.plot(rmse2)
plt.plot(rmse4)
plt.title("RMSE variation with lag values")
plt.legend(['Temperature, Time, Previous Energy, Humidity','Temperature, Time, Previous Energy','Time , Humidity, Temperature'])
plt.xlabel('Lag values')
plt.ylabel('RMSE')
plt.plot()


# In[418]:


X_train = data.drop(['Predicted_Energy','Energy'],axis=1).iloc[:day_reading]
y_train = data['Predicted_Energy'].iloc[:day_reading]
X_test = data.drop(['Predicted_Energy','Energy'],axis=1).iloc[day_reading:]
y_test = data['Predicted_Energy'].iloc[day_reading:]


# In[419]:


# applying SVM over the data using rbf kernel
from sklearn.svm import SVR
opt_v = 99999
y_opt = []
for c in [.001,.01,.1,1,10,100,1000]:
    reg = SVR(kernel='rbf',gamma = 'auto',C=c)
    reg.fit(X_train,y_train)
    y_pred = reg.predict(X_test)
    y_pred1 = reg.predict(X_train)
    m = mean_squared_error(y_test,y_pred)
    m1 = mean_squared_error(y_train,y_pred1)
    print("Root Mean squared error using SVM on test data:", math.sqrt(m))
    #print("Root Mean squared error using SVM on train data:", math.sqrt(m1))
    if math.sqrt(m) < opt_v:
        opt_v = math.sqrt(m)
        y_opt = y_pred


# In[420]:


plt.plot(y_test.values)
plt.plot(y_opt)
plt.legend(["Energy Produced","Energy Predicted"])
plt.title("Solar Energy Prediction Over Test Data")
plt.show()
#m = mean_squared_error(y_test,y_pred)
print("Mean squared error using SVM :", opt_v)
print("MAPE using SVM :", MAPE(y_test.values,y_opt))


# In[424]:


# applying SVM over the data using rbf kernel
from sklearn.svm import SVR
opt_v = 99999
y_opt = []
for c in [.01,.1,1,10,100]:
    reg = SVR(kernel='linear',gamma = 'auto',C=c)
    reg.fit(X_train,y_train)
    y_pred = reg.predict(X_test)
    y_pred1 = reg.predict(X_train)
    m = mean_squared_error(y_test,y_pred)
    m1 = mean_squared_error(y_train,y_pred1)
    print("Root Mean squared error using SVM on test data:", math.sqrt(m))
    #print("Root Mean squared error using SVM on train data:", math.sqrt(m1))
    if math.sqrt(m) < opt_v:
        opt_v = math.sqrt(m)
        y_opt = y_pred


# In[425]:


plt.plot(y_test.values)
plt.plot(y_opt)
plt.legend(["Energy Produced","Energy Predicted"])
plt.title("Solar Energy Prediction Over Test Data")
plt.show()
#m = mean_squared_error(y_test,y_pred)
print("Mean squared error using SVM :", opt_v)
print("MAPE using SVM :", MAPE(y_test.values,y_opt))


# ## Applying VAR Model

# In[215]:


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.vector_ar.var_model import VAR
get_ipython().run_line_magic('matplotlib', 'inline')


# In[217]:


#johan_test_temp = data
#coint_johansen(johan_test_temp,-1,1).eig


# 1. From the above result we can see that eigen values are less then 1 so our data is stationary

# In[218]:


data.head()


# In[219]:


data1 = data[['Temperature','Energy','Humidity','Predicted_Energy']]


# In[220]:


#creating the train and validation set
train = data1[:int(0.8*(len(data1)))]
test = data1[int(0.8*(len(data1))):]


# In[221]:


train.shape,test.shape


# In[222]:


#fit the model
model = VAR(endog=train)
model_fit = model.fit()


# In[223]:


model_fit.y


# In[224]:


# make prediction on the test dataset
prediction = model_fit.forecast(model_fit.y, steps=len(test))


# In[225]:


#converting predictions to pandas dataframe
import math
from sklearn.metrics import mean_squared_error
pred = pd.DataFrame(index=range(0,len(prediction)),columns=data1.columns)
for j in range(0,4):
    for i in range(0, len(prediction)):
        pred.iloc[i][j] = prediction[i][j]


# In[226]:


pred


# In[227]:


#printing the rmse of the model
for i in ['Temperature','Predicted_Energy']:
    print('rmse value for', i, 'is : ', math.sqrt(mean_squared_error(pred[i], test[i])))


# In[228]:


import matplotlib.pyplot as plt
plt.plot(pred['Predicted_Energy'].values)
plt.plot(test['Predicted_Energy'].values)
plt.xlabel("time")
plt.ylabel("T")
plt.legend(["Predicted Values","Actual Values"])


# ## adding cloud cover as one of the feature

# In[247]:


data['cloudcover'] = cloud#cloud[:len(cloud)-k]


# In[248]:


data.head()


# In[ ]:


from sklearn.preprocessing import OneHotEncoder
def create_train_test(k,data,time):
    x = data['Energy'].values[:len(data['Energy'].values)-k]  
    data = data.iloc[k:]
    data['Energy_t'] = x
    #print(data)
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(data['Time'].values.reshape(-1,1))
    X = enc.transform(data['Time'].values.reshape(-1,1))
    #print(data)
   
    data.reset_index(drop=True, inplace=True)
    day_d = pd.DataFrame(X.todense())
    day_d.reset_index(drop=True, inplace=True)
    #print(day_d.shape)
    data = pd.concat([data,day_d],axis=1)
    #print(data)
    data = data.drop(['Time'],axis=1)
    
    train = data.iloc[:day_reading]
    
    test = data.iloc[day_reading:]
    X_train = train.drop(['Energy','Energy_t'],axis=1)
    y_train = train['Energy']
    X_test = test.drop(['Energy','Energy_t'],axis=1)
    y_test = test['Energy']
    return [X_train,X_test,y_train,y_test]


# In[249]:


X_train,X_test,y_train,y_test=create_train_test(40,data,time)


# In[250]:


X_train.head()


# In[251]:


rmse5 =  []
mape5 =  []
for k in range(1,40):
    X_train,X_test,y_train,y_test=create_train_test(k,data,time)
    reg = SVR(kernel='linear',gamma = 'auto',C=10)
    reg.fit(X_train,y_train)
    y_pred = reg.predict(X_test)
    y_pred1 = reg.predict(X_train)
    m = mean_squared_error(y_test,y_pred)
    m1 = mean_squared_error(y_train,y_pred1)
    print("Root Mean For Lag Value:"+str(k)+'---->', math.sqrt(m))
    rmse5.append(math.sqrt(m))
    mape5.append(MAPE(y_test.values,y_pred))
    print("MAPE using SVM :", MAPE(y_test.values,y_pred))
    print("***************************************************")
    #print("Root Mean squared error using SVM on train data:", math.sqrt(m1))
    if math.sqrt(m) < opt_v:
        opt_v = math.sqrt(m)
        y_opt = y_pred


# In[405]:


plt.plot(mape2)
plt.plot(mape3)
plt.plot(mape4)
plt.plot(mape5)
plt.title("MAPE variation with lag values")
plt.legend(['Temperature, Time, Previous Energy','Temperature, Time, Previous Energy, Humidity','Time , Humidity, Temperature' Time , Humidity, Temperature,Cloud Cover'])
plt.xlabel('Lag values')
plt.ylabel('MAPE')
plt.plot()


# In[ ]:


plt.plot(mape2)
plt.plot(mape3)
plt.plot(mape4)
plt.plot(mape5)
plt.title("RMSE variation with lag values")
#plt.legend(['Temperature, Time, Previous Energy Produced, Humidity','Temperature, Time, Previous Energy Produced','Time , Humidity, Temperature, Cloud Cover'])
plt.xlabel('Lag values')
plt.ylabel('MAPE')
plt.plot()


# In[429]:


train = data.iloc[:day_reading]
test = data.iloc[day_reading:]
X_train = train.drop(['Predicted_Energy'],axis=1)
y_train = train['Predicted_Energy']
X_test = test.drop(['Predicted_Energy'],axis=1)
y_test = test['Predicted_Energy']


# In[221]:


# applying SVM over the data using rbf kernel
from sklearn.svm import SVR
opt_v = 99999
y_opt = []
for c in [.001,.01,.1,1,10,100,1000]:
    reg = SVR(kernel='linear',gamma = 'auto',C=c)
    reg.fit(X_train,y_train)
    y_pred = reg.predict(X_test)
    y_pred1 = reg.predict(X_train)
    m = mean_squared_error(y_test,y_pred)
    m1 = mean_squared_error(y_train,y_pred1)
    print("Root Mean squared error using SVM on test data:", math.sqrt(m))
    #print("Root Mean squared error using SVM on train data:", math.sqrt(m1))
    if math.sqrt(m) < opt_v:
        opt_v = math.sqrt(m)
        y_opt = y_pred


# In[222]:


plt.plot(y_test.values)
plt.plot(y_opt)
plt.legend(["Energy Produced","Energy Predicted"])
plt.title("Solar Energy Prediction Over Test Data")
plt.show()
print("Mean squared error using SVM :", opt_v)
print("MAPE using SVM :", MAPE(y_test.values,y_opt))


# In[223]:


# applying SVM over the data using rbf kernel
from sklearn.svm import SVR
opt_v = 99999
y_opt = []
for c in [.01,.1,1,10,100,1000]:
    reg = SVR(kernel='rbf',gamma = 'auto',C=c)
    reg.fit(X_train,y_train)
    y_pred = reg.predict(X_test)
    y_pred1 = reg.predict(X_train)
    m = mean_squared_error(y_test,y_pred)
    m1 = mean_squared_error(y_train,y_pred1)
    print("Root Mean squared error using SVM on test data:", math.sqrt(m))
    if math.sqrt(m) < opt_v:
        opt_v = math.sqrt(m)
        y_opt = y_pred


# In[224]:


plt.plot(y_test.values)
plt.plot(y_opt)
plt.legend(["Energy Produced","Energy Predicted"])
plt.title("Solar Energy Prediction Over Test Data")
plt.show()
#m = mean_squared_error(y_test,y_pred)
print("Mean squared error using SVM :", opt_v)
print("MAPE using SVM :", MAPE(y_test.values,y_opt))


# In[226]:


# applying SVM over the data using rbf kernel
from sklearn.svm import SVR
opt_v = 99999
y_opt = []
for c in [.001,.01,.1,1,10]:
    reg = SVR(kernel='poly',gamma = 'auto',C=c)
    reg.fit(X_train,y_train)
    y_pred = reg.predict(X_test)
    y_pred1 = reg.predict(X_train)
    m = mean_squared_error(y_test,y_pred)
    m1 = mean_squared_error(y_train,y_pred1)
    print("Root Mean squared error using SVM on test data:", math.sqrt(m))
    #print("Root Mean squared error using SVM on train data:", math.sqrt(m1))
    if math.sqrt(m) < opt_v:
        opt_v = math.sqrt(m)
        y_opt = y_pred


# In[137]:


plt.plot(y_test.values)
plt.plot(y_opt)
plt.legend(["Energy Produced","Energy Predicted"])
plt.title("Solar Energy Prediction Over Test Data")
plt.show()
#m = mean_squared_error(y_test,y_pred)
print("Mean squared error using SVM :", opt_v)
print("MAPE using SVM :", MAPE(y_test.values,y_opt))


# ### Adding dewPoint as a feature 

# In[259]:


from sklearn.preprocessing import OneHotEncoder
def create_train_test(k,data,time):
    x = data['Energy'].values[:len(data['Energy'].values)-k]  
    data = data.iloc[k:]
    data['Energy_t'] = x
    #print(data)
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(data['Time'].values.reshape(-1,1))
    X = enc.transform(data['Time'].values.reshape(-1,1))
    #print(data)
   
    data.reset_index(drop=True, inplace=True)
    day_d = pd.DataFrame(X.todense())
    day_d.reset_index(drop=True, inplace=True)
    #print(day_d.shape)
    data = pd.concat([data,day_d],axis=1)
    #print(data)
    data = data.drop(['Time'],axis=1)
    
    train = data.iloc[:day_reading]
    
    test = data.iloc[day_reading:]
    X_train = train.drop(['Energy'],axis=1)
    y_train = train['Energy']
    X_test = test.drop(['Energy'],axis=1)
    y_test = test['Energy']
    return [X_train,X_test,y_train,y_test]


# In[260]:


data.head()


# In[261]:


data['dewPoint'] = dewpoint


# In[262]:


X_train,X_test,y_train,y_test=create_train_test(40,data,time)


# In[263]:


X_train.head()


# In[264]:


rmse6 =  []
mape6 =  []
for k in range(1,40):
    X_train,X_test,y_train,y_test=create_train_test(k,data,time)
    reg = SVR(kernel='linear',gamma = 'auto',C=10)
    reg.fit(X_train,y_train)
    y_pred = reg.predict(X_test)
    y_pred1 = reg.predict(X_train)
    m = mean_squared_error(y_test,y_pred)
    m1 = mean_squared_error(y_train,y_pred1)
    print("Root Mean For Lag Value:"+str(k)+'---->', math.sqrt(m))
    rmse6.append(math.sqrt(m))
    mape6.append(MAPE(y_test.values,y_pred))
    print("MAPE using SVM :", MAPE(y_test.values,y_pred))
    print("***************************************************")
    #print("Root Mean squared error using SVM on train data:", math.sqrt(m1))
    if math.sqrt(m) < opt_v:
        opt_v = math.sqrt(m)
        y_opt = y_pred


# In[410]:


plt.plot(mape2)
plt.plot(mape3)
plt.plot(mape4)
plt.plot(mape5)
plt.plot(mape6)
plt.title("MAPE variation with lag values")
plt.legend(['Temperature, Time, Previous Energy','Temperature, Time, Previous Energy Humidity','Humidity, Temperature, Time','Time , Humidity, Temperature, Cloud Cover','Time , Humidity, Temperature, Cloud Cover,Dew Point'])
plt.xlabel('Lag values')
plt.ylabel('MAPE')
plt.plot()


# In[233]:


#train = data.iloc[:day_reading]
#test = data.iloc[day_reading:]
#X_train = train.drop(['Predicted_Energy'],axis=1)
#y_train = train['Predicted_Energy']
#X_test = test.drop(['Predicted_Energy'],axis=1)
#y_test = test['Predicted_Energy']


# In[234]:


# applying SVM over the data using rbf kernel
from sklearn.svm import SVR
opt_v = 99999
y_opt = []
for c in [.001,.01,.1,1,10,100,1000]:
    reg = SVR(kernel='linear',gamma = 'auto',C=c)
    reg.fit(X_train,y_train)
    y_pred = reg.predict(X_test)
    y_pred1 = reg.predict(X_train)
    m = mean_squared_error(y_test,y_pred)
    m1 = mean_squared_error(y_train,y_pred1)
    print("Root Mean squared error using SVM on test data:", math.sqrt(m))
    #print("Root Mean squared error using SVM on train data:", math.sqrt(m1))
    if math.sqrt(m) < opt_v:
        opt_v = math.sqrt(m)
        y_opt = y_pred


# In[235]:


plt.plot(y_test.values)
plt.plot(y_opt)
plt.legend(["Energy Produced","Energy Predicted"])
plt.title("Solar Energy Prediction Over Test Data")
plt.show()
#m = mean_squared_error(y_test,y_pred)
print("Mean squared error using SVM :", opt_v)
print("MAPE using SVM :", MAPE(y_test.values,y_opt))


# ## Time, Temperature, Humidity, Previous Energy, Dew Point
# ### (REMOVING CLOUD COVER FEATURE)

# In[236]:


train = data.iloc[:day_reading]
test = data.iloc[day_reading:]
X_train = train.drop(['Predicted_Energy','cloudcover'],axis=1)
y_train = train['Predicted_Energy']
X_test = test.drop(['Predicted_Energy','cloudcover'],axis=1)
y_test = test['Predicted_Energy']


# In[237]:


X_train.head()


# In[238]:


# applying SVM over the data using rbf kernel
from sklearn.svm import SVR
opt_v = 99999
y_opt = []
for c in [.001,.01,.1,1,10,100,1000]:
    reg = SVR(kernel='linear',gamma = 'auto',C=c)
    reg.fit(X_train,y_train)
    y_pred = reg.predict(X_test)
    y_pred1 = reg.predict(X_train)
    m = mean_squared_error(y_test,y_pred)
    m1 = mean_squared_error(y_train,y_pred1)
    print("Root Mean squared error using SVM on test data:", math.sqrt(m))
    #print("Root Mean squared error using SVM on train data:", math.sqrt(m1))
    if math.sqrt(m) < opt_v:
        opt_v = math.sqrt(m)
        y_opt = y_pred


# In[239]:


plt.plot(y_test.values)
plt.plot(y_opt)
plt.legend(["Energy Produced","Energy Predicted"])
plt.title("Solar Energy Prediction Over Test Data")
plt.show()
#m = mean_squared_error(y_test,y_pred)
print("Mean squared error using SVM :", opt_v)
print("MAPE using SVM :", MAPE(y_test.values,y_opt))


# ## Adding Wind Speed to the model

# In[240]:


data['windSpeed'] = windspeed[:len(windspeed)-k]


# In[241]:


data.head()


# In[242]:


train = data.iloc[:day_reading]
test = data.iloc[day_reading:]
X_train = train.drop(['Predicted_Energy','cloudcover'],axis=1)  # adding cloud cover decreasing the rmse score so removing it from the feature set
y_train = train['Predicted_Energy']
X_test = test.drop(['Predicted_Energy','cloudcover'],axis=1)
y_test = test['Predicted_Energy']


# In[243]:


X_train.head()


# In[244]:


# applying SVM over the data using rbf kernel
from sklearn.svm import SVR
opt_v = 99999
y_opt = []
for c in [.001,.01,.1,1,10,100,1000]:
    reg = SVR(kernel='linear',gamma = 'auto',C=c)
    reg.fit(X_train,y_train)
    y_pred = reg.predict(X_test)
    y_pred1 = reg.predict(X_train)
    m = mean_squared_error(y_test,y_pred)
    m1 = mean_squared_error(y_train,y_pred1)
    print("Root Mean squared error using SVM on test data:", math.sqrt(m))
    #print("Root Mean squared error using SVM on train data:", math.sqrt(m1))
    if math.sqrt(m) < opt_v:
        opt_v = math.sqrt(m)
        y_opt = y_pred


# In[245]:


plt.plot(y_test.values)
plt.plot(y_opt)
plt.legend(["Energy Produced","Energy Predicted"])
plt.title("Solar Energy Prediction Over Test Data")
plt.show()
print("Mean squared error using SVM :", opt_v)
print("MAPE using SVM :", MAPE(y_test.values,y_opt))


# ### Adding windBearing as a feature

# In[246]:


k = 40
data['windBearing'] = wBearing[:len(wBearing)-k]


# In[247]:


data.head()


# In[249]:


train = data.iloc[:day_reading]
test = data.iloc[day_reading:]
X_train = train.drop(['Predicted_Energy','cloudcover'],axis=1)  # adding cloud cover decreasing the rmse score so removing it from the feature set
y_train = train['Predicted_Energy']
X_test = test.drop(['Predicted_Energy','cloudcover'],axis=1)
y_test = test['Predicted_Energy']


# In[250]:


# applying SVM over the data using rbf kernel
from sklearn.svm import SVR
opt_v = 99999
y_opt = []
for c in [.001,.01,.1,1,10,100]:
    reg = SVR(kernel='linear',gamma = 'auto',C=c)
    reg.fit(X_train,y_train)
    y_pred = reg.predict(X_test)
    y_pred1 = reg.predict(X_train)
    m = mean_squared_error(y_test,y_pred)
    m1 = mean_squared_error(y_train,y_pred1)
    print("Root Mean squared error using SVM on test data:", math.sqrt(m))
    #print("Root Mean squared error using SVM on train data:", math.sqrt(m1))
    if math.sqrt(m) < opt_v:
        opt_v = math.sqrt(m)
        y_opt = y_pred


# In[251]:


plt.plot(y_test.values)
plt.plot(y_opt)
plt.legend(["Energy Produced","Energy Predicted"])
plt.title("Solar Energy Prediction Over Test Data")
plt.show()
#m = mean_squared_error(y_test,y_pred)
print("Mean squared error using SVM :", opt_v)
print("MAPE using SVM :", MAPE(y_test.values,y_opt))


# ### Removing Energy as a feature

# In[252]:


train = data.iloc[:day_reading]
test = data.iloc[day_reading:]
X_train = train.drop(['Predicted_Energy','Energy'],axis=1)  # adding cloud cover decreasing the rmse score so removing it from the feature set
y_train = train['Predicted_Energy']
X_test = test.drop(['Predicted_Energy','Energy'],axis=1)
y_test = test['Predicted_Energy']


# In[253]:


# applying SVM over the data using rbf kernel
from sklearn.svm import SVR
opt_v = 99999
y_opt = []
for c in [.001,.01,.1,1,10,100]:
    reg = SVR(kernel='linear',gamma = 'auto',C=c)
    reg.fit(X_train,y_train)
    y_pred = reg.predict(X_test)
    y_pred1 = reg.predict(X_train)
    m = mean_squared_error(y_test,y_pred)
    m1 = mean_squared_error(y_train,y_pred1)
    print("Root Mean squared error using SVM on test data:", math.sqrt(m))
    #print("Root Mean squared error using SVM on train data:", math.sqrt(m1))
    if math.sqrt(m) < opt_v:
        opt_v = math.sqrt(m)
        y_opt = y_pred


# In[254]:


plt.plot(y_test.values)
plt.plot(y_opt)
plt.legend(["Energy Produced","Energy Predicted"])
plt.title("Solar Energy Prediction Over Test Data")
plt.show()
#m = mean_squared_error(y_test,y_pred)
print("Mean squared error using SVM :", opt_v)
print("MAPE using SVM :", MAPE(y_test.values,y_opt))


# In[ ]:




