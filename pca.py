import numpy as np;
import pandas as pd;
import datetime as dt;
import matplotlib.pyplot as plt;
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri;
from workalendar.asia import Singapore;
from rpy2.rinterface import RRuntimeError;


cal=Singapore()
def processdata(filepath):
    try:
        SGX_df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
        tradedate = SGX_df['date'].unique()[0]
        date=pd.to_datetime(tradedate).date()
        if cal.is_working_day(date)==False:
            print('today is not business day:',tradedate)
            return None

        X = SGX_df[['date','series','sequence_number','change_reason','quantity_difference','price']]
        X=X[X['change_reason']==3]
        dateindex = pd.to_datetime(X.index, format='%Y-%m-%dD%H:%M:%S.%f')
        X.index = dateindex
        start = pd.to_datetime(tradedate + '08:30:00.0', format='%Y-%m-%d%H:%M:%S.%f')
        end = pd.to_datetime(tradedate + '17:00:00.0', format='%Y-%m-%d%H:%M:%S.%f')

        Underlying=X['series'].unique()
        print(Underlying)
        for i,u in enumerate(Underlying):
            if i==0:
                A=X[X['series']==u]['price'].resample('T').last()
                V=X[X['series']==u]['quantity_difference'].resample('T').sum()
            else:
                a=X[X['series']==u]['price'].resample('T').last()
                v=X[X['series']==u]['quantity_difference'].resample('T').sum()
                A=pd.concat([A,a],axis=1)
                V=pd.concat([V,v],axis=1)
                print(a.size)
        p=A[start:end]
        v=V[start:end]
        print(len(A.shape))
        # Get monthly continuous 30s-frequency data by perpetual method( weighted by volumn in each 30s)
        if len(A.shape)==1:
            p=p.fillna(method='pad')
            return np.array(p)
        else:
            vol=v.fillna(0).values
            s=np.sum(vol,1)
            s[s==0]=np.nan
            s=s.reshape(-1,1)
            v_ratio=vol/s

            p=p.fillna(method='pad')
            p=p.fillna(0)
            new_p=np.sum(p*v_ratio,axis=1)
            new_p=new_p.fillna(method='pad')
            return np.array(new_p)
    except IOError:
        print('file does not exist',filepath)


dir2 = '/Users/czq/Desktop/paper/FX_Futures_Data_Temp/'
dir1 = '/Users/czq/Desktop/paper/Data_Temp/'
name2 = 'FX_Futures_'
name1 = 'SG_Futures_'
X = []
for i in np.arange(0, 12):
    n = int(i / 12)
    m = 0.01 * (i % 12)
    s = "%.2f" % (2014.01 + n + m)
    x = []
    for j in np.arange(0, 31):
        nn = 101 + j
        ss = dir1 + name1 + s + '/' + name1 + s + '.' + str(nn)[1:] + '.csv'
        # ss=dir2+name2+s+'/'+name2+s+'.'+str(nn)[1:]+'.csv'
        print(ss)
        p = processdata(ss)
        if p is None:
            continue
        else:
            x = np.append(x,np.log(p))
            print("%s completed" % ss)

    X.append(x)

month_data=pd.DataFrame(X)
month_data.dropna(axis=1,inplace=True)

da_diff=np.diff(month_data,axis=1)
Sigma=np.cov(da_diff)
eig=np.linalg.eig(Sigma)
eigvl=eig[0]
ratio=eigvl/np.sum(eigvl)
ratio