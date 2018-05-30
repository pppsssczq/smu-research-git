import numpy as np;
import pandas as pd;
import datetime as dt;
import matplotlib.pyplot as plt;
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri;
from workalendar.asia import Singapore;
from rpy2.rinterface import RRuntimeError;

#process data keep columns=['quantity difference','change reason','price','mp_quantity','bid or ask','best price','best size']
#return 4 data, 1: the origin data at trading time, 2: is the data for running GMM, 3:underlying name, 4:trade date
cal=Singapore()
def processdata(filepath):
    try:
        SGX_df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
        tradedate = SGX_df['date'].unique()[0]
        date=pd.to_datetime(tradedate).date()
        if cal.is_working_day(date)==False:
            print('today is not business day:',tradedate)
            return None

        X = SGX_df.drop(columns=['date', 'order_number', 'ob_position', 'ob_command','mp_quantity','bestsize'])
        Uderlying=X['series'].unique()
        Re=[]
        for underlying in Uderlying:
        #underlying=x['series'][0]
            x=X[X['series']==underlying]
            '''
            if len(x[x['change_reason']==3])<=50:
                print(underlying,'amount traded order less than 50 ')
                yield [],[],underlying,date
                continue
            '''
            x=x.sort_values(by=['sequence_number'])
            x=x.drop(columns='series')
            dateindex = pd.to_datetime(x.index, format='%Y-%m-%dD%H:%M:%S.%f')
            x.index=dateindex

            start=pd.to_datetime(tradedate+'08:30:0.0',format='%Y-%m-%d%H:%M:%S.%f')
            end=pd.to_datetime(tradedate+'17:0:0.0',format='%Y-%m-%d%H:%M:%S.%f')
            lastask,lastbid = 0,0
            #'data' is for origin data,'trading' is for gmm data
            data,trading,tstamp,tstamp2=[],[],[],[]
            tstamp_1 = dt.datetime.now()
            vol=0
            for i, row in enumerate(x.values):

                timestamp = x.index[i]
                if timestamp < start:
                    if row[4] == 1:
                        lastbid = row[5]
                    elif row[4] == 2:
                        lastask = row[5]
                    continue
                elif timestamp >= start and timestamp < end:
                    tstamp2.append(timestamp)
                    if row[4] == 1:
                        if row[2] == 3:
                            if row[3] <= lastbid:
                                d = -1
                            elif row[3] == (lastbid + lastask) * 0.5:
                                d = 0
                            elif row[3] > lastbid and row[3] < (lastbid + lastask) * 0.5:
                                d = -1
                            trading.append([row[3], d])
                            tstamp.append(timestamp)
                            vol=vol+row[1]
                        lastbid = row[5]
                        data.append([row[1], row[2], row[3], row[4], lastask, lastbid])
                    elif row[4] == 2:
                        if row[2] == 3:
                            if row[3] >= lastask:
                                d = 1
                            elif row[3] == (lastbid + lastask) * 0.5:
                                d = 0
                            elif row[3] < lastask and row[3] > (lastbid + lastask) * 0.5:
                                d = 1
                            trading.append([row[3], d])
                            tstamp.append(timestamp)
                            vol=vol+row[1]
                        lastask = row[5]
                        data.append([row[1], row[2], row[3], row[4], lastask, lastbid])
                    continue
                #for the reason the origin data is a little mess, we delete the data in last part whose sequence number is not in order
                elif timestamp>=end:
                    break
            tstamp_2 = dt.datetime.now()

            re_gmm=pd.DataFrame(trading,index=tstamp,columns=['price','direction'])
            re_origin=pd.DataFrame(data,index=tstamp2,columns=['quantity_difference','change_reason','price','bid_or_ask','bestask','bestbid'])
            print('Time taken for process: %d' % (tstamp_2 - tstamp_1).seconds)
            #return 4 data, first is the original data, second is the data for gmm, the third is underlying name, last is the date
            yield re_origin, re_gmm, underlying,date,vol

    except IOError:
        print("the file doesn't exist: ",filepath)
        return None
    except IndexError:
        print('the file is abnormal',filepath)
        r=pd.DataFrame([])
        return r,r

#computer auto covariance ,order is the interval, Eg, order=2 means autocovariance between Xt and Xt-2
def autocov(x, order):
    x = np.array(x)
    u = np.mean(x)
    r = (x[order:] - u) * (x[:-order] - u)
    return r.mean()

#compute effective spread according to ROLL's Model
def effective_spread(price):
    price_diff = price.diff()[1:]
    autocov_price_diff = autocov(price_diff, 1)
    effective_spread = 2 * (-autocov_price_diff) ** 0.5
    print("the effective spread is: ", effective_spread)
    return effective_spread

def herf_index(data):
    s=[]
    date=data[0][3]
    for d in data:
        d=d[0]
        d=d[d['change_reason']==3].values
        v=np.sum(d[:,0])
        s.append(v)
    r=np.array(s[:2])/np.sum(s)
    return date,r

def order_split(data,freq):
    r,time=[],[]
    index=data.index
    for i,j in enumerate(data.values):
        if i==0:
            r.append(j)
            time.append(index[i])
            tempt=j
        else:
            #if direction same, and if price same then consider as one order
            if j[1]==tempt[1]:
                if j[0]==tempt[0]:
                    continue
                else:
                    r.append(j)
                    time.append(index[i])
                    tempt = j
            else:
                r.append(j)
                time.append(index[i])
                tempt=j

    x=pd.DataFrame(r,columns=['price','direction'],index=time)
    if freq:
        x=x.resample(freq).last()
        x=x.dropna()

    return x

def rungmm(data,freq):
    try:
        tstamp_1 = dt.datetime.now()
        data_gmm=data
        if freq:
            data_gmm=data_gmm.resample(freq).last()
            data_gmm=data_gmm.dropna()
        #data_gmm=data_gmm.groupby(level=0).last()
        print(len(data))
        #data_gmm=order_split(data,freq)
        #if len(data_gmm)<100:
        #    return [np.nan,np.nan,np.nan]
        size=len(data_gmm)
        print(len(data_gmm))
        p = 10000 * np.log(data_gmm['price'])
        d = data_gmm['direction']
        mean=np.mean(p)
        table=np.stack((d[:-1],d[1:],p[:-1],p[1:]),axis=1)
        table = pd.DataFrame(table,columns=['xt0', 'xt1', 'pt0', 'pt1'])

        # para: [theta, phi, rho, lamda]
        pandas2ri.activate()
        robjects.r('library(sandwich)')
        robjects.r('library(gmm)')
        robjects.r('''
        g1 <- function(tet, x)
        {   
            u <- (1.0*(x[,4] - x[,3]) - (tet[2] + tet[1])*x[,2] +
                  (tet[2] + tet[3]*tet[1])*x[,1])
            a <- mean(u)
            m1 <- (x[,1]*x[,2] - tet[3]*x[,2]^2)
            m2 <- u-a
            m3 <- ((u-a) * x[,1])
            m4 <- ((u-a) * x[,2])
            m5 <- (abs(x[,1])+tet[4]-1)
            f <- cbind(m1, m2, m3, m4, m5)
            return(f)
        }
        ''')
        r_table = pandas2ri.py2ri(table)
        g1 = robjects.globalenv['g1']
        robjects.r('library(sandwich)')
        robjects.r('library(gmm)')
        r_gmm = robjects.r['gmm']
        r_matrix = robjects.r['data.matrix']
        r_summary = robjects.r['summary']
        r_coef = robjects.r['coef']
        c = robjects.r['c']
        res = r_gmm(g1, r_matrix(r_table), c(theta=0.1, phi=0.1, rho=0.1,lamda=0.1))
        tstamp_2 = dt.datetime.now()

        print('Time taken for GMM calculation: %d' % (tstamp_2 - tstamp_1).seconds)
        print(res)
        res_summary = r_summary(res)
        res_coeff = r_coef(res_summary)
        print(res_coeff)
        coeff=[res_coeff[0],res_coeff[1],res_coeff[2],res_coeff[3],size]


        return coeff
    except RRuntimeError:
        return[np.nan,np.nan,np.nan,np.nan,size]

def Herf_index(dataset):
    pass

def RV(x,freq):
    if freq:
        x=x.resample(freq).last()
        x=x.dropna()
    x=100*np.log(x.values[:,0])
    if len(x)<10:
        return None
    else:
        return np.sum(np.diff(x)**2)

N=[]

dir2='/Users/czq/Desktop/paper/FX_Futures_Data_Temp/'
dir1='/Users/czq/Desktop/paper/Data_Temp/'
name2='FX_Futures_'
name1='SG_Futures_'
series,abnormal,Date,other=[],[],[],[]
#automatically reading all the files starting from 2014.01
re_freq=[]
re_freq=[[] for i in range(5)]
series=[[] for i in range(5)]
Date=[[] for i in range(5)]
realized_var=[[] for i in range(5)]
freq=[0,'s','10s','30s','T']
Herf,Herf_date=[],[]

for i in np.arange(0,24):
    n=int(i/12)
    m=0.01*(i%12)
    s="%.2f"%(2014.01+n+m)
    for j in np.arange(0,31):
        nn=101+j
        ss=dir1+name1+s+'/'+name1+s+'.'+str(nn)[1:]+'.csv'
        #ss=dir2+name2+s+'/'+name2+s+'.'+str(nn)[1:]+'.csv'
        print(ss)
        intraday=processdata(ss)

        if(intraday==None):
            continue
        else:

            for d in intraday:
                d_gmm=d[1]
                if len(d_gmm)==0:
                    print('no trading data on this day')
                    abnormal.append(ss)
                    continue
                print(d[3],d[2],len(d[1]))
                other.append([d[3],d[2],len(d[1])])
                for t,f in enumerate(freq):
                    g_gmmresult=rungmm(d_gmm,f)
                    re_freq[t].append(g_gmmresult)
                    series[t].append(d[2])
                    Date[t].append(d[3])


Myre=[]
title=['origin','1s','10s','30s','1T']
writer=pd.ExcelWriter('para_rv.xlsx')
for i in range(5):
    mr=pd.DataFrame(re_freq[i],columns=['theta','phi','rho','lamda','size'])
    sr=pd.Series(series[i])
    dt=pd.Series(Date[i])
    myre=pd.concat([sr,dt,mr],axis=1)
    Myre.append(myre)
    myre.to_excel(writer,title[i])
writer.save()




'''
intraday=processdata(dir+name+'2015.12/'+'SG_Futures_2015.12.22.csv')

for d in intraday:
    try:
        d_gmm=d[1]
        if len(d_gmm)==0:
            print('no trading data on this day')
            #abnormal.append(ss)
            continue
        print(d[3],d[2],len(d[1]))
        print(d[1])
        g_gmmresult=rungmm(d_gmm,'T')
        N.append(g_gmmresult)
        series.append(d[2])
        Date.append(d[3])
    except rpy2.rinterface.RRuntimeError:
        continue

'''










































