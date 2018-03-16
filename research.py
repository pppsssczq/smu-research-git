import numpy as np;
import pandas as pd;
import datetime as dt;
import matplotlib.pyplot as plt;
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri;
from scipy.optimize import least_squares


#process data keep columns=['quantity difference','change reason','price','mp_quantity','bid or ask','best price','best size']
def processdata(filepath):
    SGX_df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
    ad_SGX = SGX_df.drop(columns=['date', 'series', 'sequence_number', 'order_number', 'ob_position', 'ob_command'])

    new = ad_SGX['2014-01-02D08:30:00.045388600':'2014-01-02D16:59:59.890442600']
    new.index = pd.to_datetime(new.index, format='%Y-%m-%dD%H:%M:%S.%f')
    traded = new[new['change_reason'] == 3]


    return traded

#resample data in different frequency
def resampledata(traded,freq):

    direction=traded['bid_or_ask'].groupby(level=0).last()
    direction=direction.resample(freq).last()
    direction=direction.fillna(method='pad')

    price = traded['price'].groupby(level=0).last()
    price=price.resample(freq).last()
    price = price.fillna(method='pad')

    bestprice = traded['bestprice'].groupby(level=0).mean()
    bestprice = bestprice.resample(freq).last()
    bestprice=bestprice.fillna(method='pad')

    traded=pd.concat([direction,price,bestprice],axis=1)

    return traded
#plot transaction price and  ask price and bid price available on sequence in different frequency
def plotprice(traded,freq):
    new1 = traded[(traded['bid_or_ask'] == 1)]  # traded bid order
    new2 = traded[(traded['bid_or_ask'] == 2)]  # traded ask order

    price=traded['price']
    price =price.resample(freq).last()
    price = price.fillna(method='pad')

    bp1 = new1['bestprice']
    bp1=bp1.resample(freq).last()
    bp1=bp1.fillna(method='pad')
    bp2 = new2['bestprice']
    bp2 = bp2.resample(freq).last()
    bp2 = bp2.fillna(method='pad')

    plt.figure(1)
    bp1.plot(linestyle='None', marker='+', markersize=2)
    bp2.plot(linestyle='None', marker='+', markersize=2)
    price.plot(linestyle='None', marker='+', markersize=2)
    plt.show()
#plot volumn in different frequency
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

    new1 = data[(data['bid_or_ask'] == 1)]# traded bid order
    new2 = data[(data['bid_or_ask'] == 2)]  # traded ask order
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
def direction(x):
    for i,j in enumerate(x):
        if j==1:
            x[i]=-1
        elif j==2:
            x[i]=1
        else:
            x[i]=0
    return x

#computer auto covariance ,order is the interval, Eg, order=2 means autocovariance between Xt and Xt-2
def autocov(x, order):
    x = np.array(x)
    u = x.mean()
    r = (x[order:] - u) * (x[:-order] - u)
    return r.mean()
#compute effective spread according to ROLL's Model
def effective_spread(price):
    price_diff = price.diff()[1:]
    autocov_price_diff = autocov(price_diff, 1)
    effective_spread = 2 * (-autocov_price_diff) ** 0.5
    print("the effective spread is: ", effective_spread)
    return effective_spread

def rungmm(d,p):
    table=np.stack((d[:-1],d[1:],p[:-1],p[1:]),axis=1)
    table = pd.DataFrame(table,columns=['xt0', 'xt1', 'pt0', 'pt1'])

    tstamp_1 = dt.datetime.now()

    # para: [theta, phi, rho]
    pandas2ri.activate()
    robjects.r('library(sandwich)')
    robjects.r('library(gmm)')

    robjects.r('''
        g1 <- function(tet, x)
        {
            u <- (1.0 * (x[,4]-x[,3])-(tet[1] + tet[2])*x[,2] + (tet[2] + tet[1] * tet[3]) * x[,1])
            m1 <- (x[,1]*x[,2]-tet[3] * x[,2] ^ 2)
            m2 <- u
            m3 <- (u*x[,1])
            m4 <- (u *x[,2])
            f <- cbind(m1, m2, m3, m4)
            return (f)
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
    res = r_gmm(g1, r_matrix(r_table), c(theta=0.1, phi=0.1, rho=0.1))
    tstamp_2 = dt.datetime.now()
    print('Time taken for GMM calculation: %d' % (tstamp_2 - tstamp_1).seconds)
    print(res)
    res_summary = r_summary(res)
    res_coeff = r_coef(res_summary)
    print(res_coeff)
    theta_p = res_coeff[9]
    phi_p = res_coeff[10]
    rho_p = res_coeff[11]
    print('theta p-values: %f, phi p-values: %f, rho p-values: %f'
          % (theta_p, phi_p, rho_p))



data=processdata('SG_Futures_2014.01.02.csv')
Redata=resampledata(data,'T')
plotvolumn(data,'T')
plotprice(data,'T')
direction=direction(data['bid_or_ask'])
price=data['price']

Effspread=effective_spread(price)

loc1=0
m=len(price)
frac=1
#by changing frac, you can seperate the one day data into several fractions.
for i in range(0,frac):
    if i==frac-1:
        loc2=m
    else:
        loc2 = int(loc1 + m / frac)

    dfrac=direction[loc1:loc2]
    pfrac=price[loc1:loc2]
    gmmresult=rungmm(dfrac,pfrac)

    loc1=loc2


















