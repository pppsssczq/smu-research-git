import numpy as np;
import pandas as pd;
import datetime as dt;
import matplotlib.pyplot as plt;
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri;
from scipy.optimize import least_squares;
from workalendar.asia import Singapore;


#process data keep columns=['quantity difference','change reason','price','mp_quantity','bid or ask','best price','best size']
#return two set of data, one is the origin data at trading time, the other is the data for running GMM
cal=Singapore()


def processdata(filepath):
    try:
        SGX_df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)

        tradedate = SGX_df['date'].unique()[0]
        date=pd.to_datetime(tradedate).date()
        if cal.is_working_day(date)==False:
            print('today is not business day:',tradedate)
            return None
        x = SGX_df.drop(columns=['date', 'order_number', 'ob_position', 'ob_command','mp_quantity','bestsize'])

        underlying=x['series'][0]
        x=x[x['series']==underlying]
        x=x.sort_values(by=['sequence_number'])
        x=x.drop(columns='series')
        dateindex = pd.to_datetime(x.index, format='%Y-%m-%dD%H:%M:%S.%f')
        x.index=dateindex

        start=pd.to_datetime(tradedate+'08:30:0.0',format='%Y-%m-%d%H:%M:%S.%f')
        end=pd.to_datetime(tradedate+'17:0:0.0',format='%Y-%m-%d%H:%M:%S.%f')
        lastask = 0
        lastbid = 0
        #data is for origin data,trading is for gmm data
        trading,tstamp,tstamp2,data=[],[],[],[]
        tstamp_1 = dt.datetime.now()
        for i,row in enumerate(x.values):

            timestamp=x.index[i]
            if timestamp < start:
                if  row[4]==1:
                    lastbid=row[5]
                elif  row[4]==2:
                    lastask=row[5]
                continue
            elif timestamp>=start and timestamp<end:
                tstamp2.append(timestamp)
                if row[4] == 1:
                    if row[2] == 3:
                        if row[3] == lastbid:
                            d = -1
                        elif row[3] > lastbid:
                            d = 0
                        trading.append([row[3], d])
                        tstamp.append(timestamp)
                    lastbid=row[5]
                    data.append([row[1], row[2], row[3], row[4], lastask,lastbid])
                elif row[4] == 2:
                    if row[2] == 3:
                        if row[3] == lastask:
                            d = 1
                        elif row[3] < lastask:
                            d = 0
                        trading.append([row[3],d])
                        tstamp.append(timestamp)
                    lastask =row[5]
                    data.append([row[1], row[2], row[3], row[4], lastask, lastbid])
                continue
            #for the reason the origin data is a little mess, we delete the data in last part whose sequence number is not in order
            elif timestamp>=end:
                break
        tstamp_2 = dt.datetime.now()

        re_gmm=pd.DataFrame(trading,index=tstamp,columns=['price','direction'])


        re_origin=pd.DataFrame(data,index=tstamp2,columns=['quantity_difference','change_reason','price','bid_or_ask','bestask','bestbid'])
        print('Time taken for process: %d' % (tstamp_2 - tstamp_1).seconds)
        #return 3 data, first is the original data, second is the data for gmm, the third is underlying name
        return re_origin, re_gmm, underlying,date
    except IOError:
        print("the file doesn't exist: ",filepath)

        return None
    except IndexError:
        print('the file is abnormal',filepath)
        r=pd.DataFrame([])
        return r,r



#resample data in different frequency
#choose the last one for price
def resampledata(traded,freq):
    traded = traded.resample(freq).last()
    traded = traded.fillna(method='pad')
    return traded
#plot transaction price and  ask price and bid price available on sequence in different frequency
def plotprice(data,freq):

    traded=data[data['change_reason']==3]
    new1 = traded[(traded['bid_or_ask'] == 1)]  # traded bid order
    new2 = traded[(traded['bid_or_ask'] == 2)]  # traded ask order

    traded=traded.resample(freq).last()
    traded=traded.fillna(method='pad')
    price=traded['price']
    bp1=traded['bestask']
    bp2=traded['bestbid']

    plt.figure(1)
    bp1.plot(linestyle='None', marker='+', markersize=2)
    bp2.plot(linestyle='None', marker='+', markersize=2)
    price.plot(linestyle='None', marker='+', markersize=2)
    plt.show()

#,sum the quantity in different frequency and plot it
#
def plotvolumn(data,freq):
    def volumn(x, y):  # bid order traded means someone sells  stocks at the bid price, which means selling
        for i in np.arange(len(x)):  # ask order traded means someone buys stocks at the ask price, which means buying
            if abs(x[i]) >= abs(y[i]):
                x[i] = abs(x[i] + y[i])
                y[i] = 0
            else:
                x[i] = 0
                y[i] = abs(x[i] + y[i])
        return x, y
    traded=data[data['change_reason']==3]
    new1 = traded[(traded['bid_or_ask'] == 1 )]# traded bid order
    new2 = traded[(traded['bid_or_ask'] == 2 )]  # traded ask order
    volumn1 = new1['quantity_difference'].resample(freq).sum()
    volumn1=volumn1.fillna(0)
    volumn2 = new2['quantity_difference'].resample(freq).sum()
    volumn2=volumn2.fillna(0)


    sell_size = volumn(volumn1, volumn2)[0]
    buy_size = volumn(volumn1, volumn2)[1]

    plt.figure(2)
    sell_size.plot(marker='+', linestyle='None', color='green', markersize=2)
    buy_size.plot(marker='+', linestyle='None', color='red', markersize=2)
    plt.show()

#transaction direction,  bid order(1) traded means someone is selling (-1), and vice versa


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

def rungmm(data_gmm,freq):
    tstamp_1 = dt.datetime.now()
    data_gmm=data_gmm.resample(freq).last()
    data_gmm=data_gmm.fillna(method='pad')
    p = 10000 * np.log(data_gmm['price'])
    d = data_gmm['direction']
    mean=np.mean(p)
    table=np.stack((d[:-1],d[1:],p[:-1],p[1:]),axis=1)
    table = pd.DataFrame(table,columns=['xt0', 'xt1', 'pt0', 'pt1'])



    # para: [theta, phi, rho]
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
    coeff=[res_coeff[0],res_coeff[1],res_coeff[2]]
    #theta_p = res_coeff[9]
    #phi_p = res_coeff[10]
    #rho_p = res_coeff[11]
    #print('theta p-values: %f, phi p-values: %f, rho p-values: %f'
       #   % (theta_p, phi_p, rho_p))
    return coeff
def RV(x):
    return np.sum(np.diff(x)**2)



Data=processdata('SG_Futures_2014.01.02.csv')
data_gmm=Data[1]
data_origin=Data[0]

#plotprice(data_origin,'T')
#plotvolumn(data_origin,'T')

price=data_gmm['price']
price=100*np.log(resampledata(price,'T'))
rv=RV(price)
#Effspread=effective_spread(price)

#gmmresult=rungmm(data_gmm,'T')

N=[]
dir='Data_Temp/'
name='SG_Futures_'
s1,s2,series,abnormal,Date=[],[],[],[],[]
#automatically reading all the files starting from 2014.01

for i in np.arange(0,24):
    n=int(i/12)
    m=0.01*(i%12)
    s="%.2f"%(2014.01+n+m)
    s1.append(s)
    for j in np.arange(0,31):
        nn=101+j
        ss=dir+name+s+'/'+name+s+'.'+str(nn)[1:]+'.csv'
        print(ss)

        d=processdata(ss)
        if(d==None):
            continue
        d_gmm=d[1]
        if len(d_gmm)==0:
            print('no trading data on this day')
            abnormal.append(ss)
            continue
        g_gmmresult=rungmm(d_gmm,'T')
        N.append(g_gmmresult)
        series.append(d[2])
        Date.append(d[3])

myre=pd.DataFrame(N,columns=['theta','phi','rho'])
myre['phi'].plot()
myre['theta'].plot()
myre['rho'].plot()
plt.legend()

Series=pd.Series(series)
DDate=pd.Series(Date)
multi=pd.concat([Series,DDate],axis=1)
Myre=pd.concat([multi,myre],axis=1)







































