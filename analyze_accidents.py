import pandas as pd 
import numpy as np
from numpy import linalg 
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.linalg
import scipy.stats
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import random
import math

acc=pd.read_csv("Accidents0514.csv")


print ("Data set shape {}".format(acc.shape))
print ('!-----------------------!')
#print acc.head()
print ('!-----------------------!')
#print acc.info()

########################

print "question 1"

#print acc['Urban_or_Rural_Area']
#print acc['Urban_or_Rural_Area'].unique()

#print acc['Urban_or_Rural_Area'].value_counts()

ll= list(acc['Urban_or_Rural_Area'].value_counts())
ll=[float(i) for i in ll]

#question 1
print ll[0]/sum(ll)

########################

print "question 2"

def roundtime(tm):
    tmr=str(tm).split(':')[0]
    return tmr

acc['Time_rounded']=acc['Time'].apply(roundtime)

#gb = acc['Accident_Severity'].groupby(acc['Time_rounded'])
#print acc['Accident_Severity']
#gb = acc.groupby('Time_rounded')['Accident_Severity']
#gb = acc.groupby(['Time_rounded', 'Accident_Severity'])['Accident_Severity'].sum()
#gb = acc.groupby(['Time_rounded','Accident_Severity'])['Accident_Severity']

#groups = dict(list(gb))
#print groups

s1 = acc[acc['Accident_Severity']==1].groupby(['Time_rounded'])['Accident_Severity'].count()
s2 = acc.groupby(['Time_rounded'])['Accident_Severity'].count()

print max(s1/s2)

########################

print "question 3"

def getyear(dt):
    dtr=str(dt).split('/')[-1]
    return dtr

acc['Year']=acc['Date'].apply(getyear)

#print acc['Year']

s3 = acc.groupby(['Year'])['Accident_Index'].count()

plx = np.asarray(s3.index,dtype=np.float)
ply = np.asarray(s3.values,dtype=np.float)

#sns.regplot(x=plx, y=ply);
#plt.show()

regr = linear_model.LinearRegression()
regr.fit(plx.reshape(-1, 1), ply)
y_pred = regr.predict(plx.reshape(-1, 1))
#print('Coefficients: \n', regr.coef_)
#print("Mean squared error: %.2f" % mean_squared_error(y_pred, ply))
#print('Variance score: %.2f' % r2_score(y_pred, ply))
print regr.coef_

########################

print "question 4"

s4 = acc[acc['Accident_Severity']==1].groupby(['Speed_limit'])['Accident_Severity'].count()
s5 = acc.groupby(['Speed_limit'])['Speed_limit'].count()
s6=s4/s5

s6=s6.fillna(0)

#print s6

v1 = np.asarray(s6.index,dtype=np.float)
v2 = np.asarray(s6.values,dtype=np.float)

print scipy.stats.pearsonr(v1,v2)[0]

########################

print "question 5"

veh=pd.read_csv("Vehicles0514.csv")
veh['Skidding_and_Overturning'] = veh['Skidding_and_Overturning'].astype(np.int)
veh['skid']=1*((veh['Skidding_and_Overturning']>=1) & (veh['Skidding_and_Overturning']<=5))


s7 = veh.groupby(['Accident_Index'])['skid'].sum()
szs7=len(s7.index)
s7.index=np.arange(szs7) 

acc['rainsnow']=acc['Number_of_Vehicles']*((acc['Weather_Conditions']==2) | (acc['Weather_Conditions']==3) | (acc['Weather_Conditions']==5) | (acc['Weather_Conditions']==6))  
acc['niceweather']=acc['Number_of_Vehicles']*(acc['Weather_Conditions']==1)

acc['rainsnow_logic']=((acc['Weather_Conditions']==2) | (acc['Weather_Conditions']==3) | (acc['Weather_Conditions']==5) | (acc['Weather_Conditions']==6))
acc['niceweather_logic']=(acc['Weather_Conditions']==1)


sumrain = acc['rainsnow'].sum()
sumnice = acc['niceweather'].sum()


#s8=pd.Series(acc['niceweather'], index=acc['Accident_Index'].values)
tt = pd.concat([acc['rainsnow_logic'], acc['niceweather_logic'], s7], axis=1)

l1 = tt.groupby(['rainsnow_logic'])['skid'].sum()/sumrain

l2 = tt.groupby(['niceweather_logic'])['skid'].sum()/sumnice

l3 = l1/l2

for i in [0, 1]:
	if (l3.index[i]): 
		print l3.values[i]

########################

print "question 6"

veh['femaled'] = 1*((veh['Sex_of_Driver']==2) & (veh['Vehicle_Type']==9))
veh['maled'] = 1*((veh['Sex_of_Driver']==1) & (veh['Vehicle_Type']==9))

smale = veh.groupby(['Accident_Index'])['maled'].sum()
sfemale = veh.groupby(['Accident_Index'])['femaled'].sum()
szm=len(smale.index)
szf=len(sfemale.index)
smale.index=np.arange(szm)
sfemale.index=np.arange(szf)

acc['sev'] = (acc['Accident_Severity'] == 1)
tfm = pd.concat([acc['sev'], smale, sfemale], axis=1)


#print tfm
rat1 = tfm.groupby(['sev'])['maled'].sum()/tfm['maled'].sum()

rat2 = tfm.groupby(['sev'])['femaled'].sum()/tfm['femaled'].sum()

ratot = rat1/rat2


for i in [0, 1]:
        if (ratot.index[i]):
                print ratot.values[i]

########################

print "Correction to question 4"

casual = pd.read_csv("Casualties0514.csv") 

scas = casual.groupby(['Accident_Index'])['Accident_Index'].count()
speedl = acc['Speed_limit']
speedl.index = acc['Accident_Index']

totspeed = pd.concat([speedl, scas], axis=1)
groupspeed = totspeed.groupby(['Speed_limit'])['Accident_Index'].sum()/totspeed.groupby(['Speed_limit'])['Accident_Index'].count()

v1 = np.asarray(groupspeed.index,dtype=np.float)
v2 = np.asarray(groupspeed.values,dtype=np.float)

print scipy.stats.pearsonr(v1,v2)[0]

########################

print "question 7"

mean_latitu = acc.groupby(['Local_Authority_(District)'])['Latitude'].mean()*(2.0*3.1415/360.0)
mean_latitu = mean_latitu.apply(np.cos)

longitu = acc.groupby(['Local_Authority_(District)'])['Longitude'].std()*(2.0*3.1415/360.0)
longitu = longitu*mean_latitu*6371.0 

latitu = acc.groupby(['Local_Authority_(District)'])['Latitude'].std()*(2.0*3.1415/360.0)*6371.0

surface = 3.1415 * longitu * latitu

print surface.max()

########################

print "question 8"

#print veh['Age_of_Driver'].min(),veh['Age_of_Driver'].max(),veh.shape
veh=veh[veh['Age_of_Driver']>=17]
#print veh['Age_of_Driver'].min(),veh.shape

veh.groupby(['Age_of_Driver'])['Age_of_Driver'].count().apply(np.log).plot()

s10 = veh.groupby(['Age_of_Driver'])['Age_of_Driver'].count().apply(np.log)

plx = np.asarray(s10.index,dtype=np.float)
ply = np.asarray(s10.values,dtype=np.float)

#sns.regplot(x=plx, y=ply);
#plt.show()

regr = linear_model.LinearRegression()
regr.fit(plx.reshape(-1, 1), ply)
print regr.coef_
y_pred = regr.predict(plx.reshape(-1, 1))
plt.plot(plx.reshape(-1, 1), y_pred, color='blue', linewidth=3)
plt.show()



