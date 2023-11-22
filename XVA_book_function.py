#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Random number part
import numpy as np
import random
def randomNumber_1(n):
    a = []
    for i in range(0, n):
        a.append(random.random())
    rn = np.sqrt(12.0/float(n)) *(np.sum(a) - n/2.0)
    return rn

def randomNumber_3():
    return np.random.randn(1)


# In[2]:


import numpy as np
n = 12

class RiskFactorModelling():
    print ('This is RiskFactorModelling')
    def __init__(self, vol, dt):
        self.vol = vol
        self.dt = dt
        self.dB = randomNumber_1(n) * np.sqrt(self.dt)

    def hestonVol(self):
        theta = 0.5  # mean reversion parameters
        w = 0.16  # long term volatility square - the target mean variance
        etha = 0.2  # vol of vol
        if 2*theta*w <= etha**2:
           raise Exception("feller condition breached for Heston model")
        variance_tminus1 = self.vol ** 2
        variance_t = variance_tminus1 + theta * (w - variance_tminus1)* self.dt + etha * np.sqrt(variance_tminus1) * self.dB
        return variance_t

    def geometricBrownian(self, s_tminus1, resetVol=False, rateCorrBM =9.99):
        self.setCorrW(rateCorrBM)
        mu = 0.05 
        self.setVol(resetVol)
        s_t =s_tminus1*np.exp((mu-self.vol**2/2)*self.dt+ self.vol*self.dB)
        return s_t

    def vasicek(self, s_tminus1, resetVol=False, rateCorrBM =9.99):
        a = 0.5  # mean reverting speed
        b = 0.04  # long term yield curve target
        self.setCorrW(rateCorrBM)
        self.setVol(resetVol)
        s_t = a *(b - s_tminus1) * self.dt + self.vol * self.dB + s_tminus1
        return s_t

    def setVol(self, resetVol):
        # this is just to set up a new vol with Heston vol
        if resetVol == True:
            self.vol = np.sqrt(self.hestonVol())
  
    def setCorrW(self, rateCorrBM):
         #dB = randomNumber_1(100) * np.sqrt(dt)
        if rateCorrBM == 9.99:
            self.dB = self.dB #print('rho default')#dW = self.dB # if the input correlation is missing, we use the instance of the class object's BM
        else:
            # if there is a rateCorrBM input, like in the cpn path simualation case, we need to create a correlated BM using the class object BM and the newly created i.i.d
            dZ = randomNumber_1(100) * np.sqrt(self.dt)
        # we create another independent Bownian motion
            self.dB = rateCorrBM * self.dB + np.sqrt(1 - rateCorrBM ** 2) * dZ


# In[3]:


def hestonVol(vol, dt):
    theta = 0.5  #mean reversion parameters
    w = 0.16  #long term volatility square - the target mean variance
    etha = 0.2 # vol of vol
    dB = randomNumber_1(100)* np.sqrt(dt)
    if 2*theta*w <= etha**2:
        raise Exception("feller condition breached for Heston model")

    variance_tmunis1 = vol**2
    variance_t = variance_tmunis1 + theta*(w - variance_tmunis1)*dt + etha*np.sqrt(variance_tmunis1)*dB
    return variance_t

def geometricBrownian(underlying_tminus1, dt, vol, rateCorrBM = 9.99, resetVol = False):
    dW = setCorrW(rateCorrBM, dt)
    mu = 0.05
    if resetVol == True:
        vol = np.sqrt(hestonVol(vol, dt))
    underlying = underlying_tminus1 * np.exp((mu - vol* vol/2)*dt + vol * dW)
    return underlying

def geometricBrownian_2(underlying_tminus1, dt, vol, rateCorrBM = 9.99, resetVol = False):
    self.setCorrW(rateCorrBM, dt)
    if resetVol == True:
        vol = np.sqrt(hestonVol(vol, dt))
    mu = 0.05
    underlying = underlying_tminus1/(1-mu *dt - vol*dW)
    return underlying

def vasicek(r_tminus1, dt, vol, rateCorrBM = 9.99, resetVol = False):
    a = 0.5  # mean reverting speed
    b = 0.02  # long term yield curve target
    self.setCorrW(rateCorrBM, dt)
    if resetVol == True:
        vol = np.sqrt(hestonVol(vol, dt))
    d_r = a * (b - r_tminus1) * dt + vol* self.dB 
    r_t = r_tminus1 + d_r
    return r_t


def setCorrW(rateCorrBM, dt):
    dB = randomNumber_1(100) * np.sqrt(dt)
    if rateCorrBM == 9.99:
        dW = dB # if the input correlation is missing as in the first riskfree rate case, we just use dB
    else: # if we do have a non 9.99 default value rateCorrBM input as in the cpn simulation case, 
        #we make a correlated dW based on the dB from the risk free case and another i.i.d dZ - but in thi simplementation, the dB
        #the dB from RF rate case changed as no Object created and the whole first memeory including the dB is erased. so we need OOP
        dZ = randomNumber_1(100) * np.sqrt(dt)
	# we create another independent Bownian motion
        dW = rateCorrBM * self.dB + np.sqrt(1 - rateCorrBM ** 2) * dZ
    return dW


# In[4]:


def payoff(row, column, cashflowNb, simInput, cpn_index_grid, T):
    cf_pv = 0.0 # initialization
    cf_pv_ftp = 0.0 # using the funding rate to discount rather than the risk free rate
    if row == 0 and column ==0:
        print ("you called Payoff function which will be for both linear (IRS) and nonlinear(IR option) pricer")
    if row in range(int(dcf * 250), timeStep+1, int(dcf * 250)):# shows that we move to the next cashflow
        cashflowNb = cashflowNb +1 
    disRate = simInput.transpose()[column][simInput.transpose()[column]!= 0].mean() 
    # by transpose , each [] we look at the column value which is not equals to zero
    #Still a question whether we should use mean value or spot
    disRate = simInput[row][column] # probably calculating the EE and df, we use row rathr than row +1
    #cpn_index = cpn_index_grid[row][column]
    # below block is for the purpose of calculating the PV which is the sum of each future expected cashflow payments
    for j in range(cashflowNb,T+1): # In order to calculate the expected future cashflow value, we need to discount back future cashflow upto the simulated timestep
        t = (j*(250*dcf)-row)/float(250)
        df = np.exp(-disRate* t) # ir or the simInput is our risk free rate, which is used as discountnig curve
        df_ftp = np.exp(-(disRate+FTPSpread) * t) # this is the FTP curve = risk free + ftp spread 
        cf_raw = (cpn_index_grid[row][column] + swapRateSpread - fixedLegCpn) * dcf * notional  # here is cf_raw is purely for periodic cashflow
        cf_pv = cf_pv + cf_raw *df # here the cf_pv is the forward pv - Expected future receivable under the forward measure
        cf_pv_ftp = cf_pv_ftp + cf_raw * df_ftp
    return cf_pv, cashflowNb, cf_pv_ftp


# In[6]:


def simulation(simStep, timeStep, sigma, dt, ir, cpn_index,PaymentTime,stochasticVol, model): # by default we use the fisr random number generatio
#def simulation():
    simInput = np.zeros((timeStep, simStep)) # this is for discount curve
    cpn_index_grid = np.zeros((timeStep, simStep)) # this is for cpn inde
    pv_grid = np.zeros((timeStep, simStep))
    pv_grid_ftp = np.zeros((timeStep, simStep))
    variance = np.zeros((timeStep, simStep))
    variance_t = sigma * sigma
    T = int(PaymentTime)
    # The Monte Carlo simualation starts
    if model not in ('GBM','vasicek' ,'nonSim'):
        raise Exception("no model mathcing, please check your model choice")
    if model == 'nonSim':
        simInput[:][:] = riskFree_ir
        cpn_index_grid[:][:] = cpn_index
        variance[:][:] = variance_t
    for column in range(0, simStep): 
        cashflowNb = 1 # we expect the first cashflow from inception t_0
        for row in range(0, timeStep):
            if row == 0 and model != 'nonSim':
                simInput[0][column] = ir
                cpn_index_grid[0][column] = cpn_index
                variance[0][column] = variance_t
                vol = sigma
                # this is to make sure each that each time a new random number is generated rather than re use what has been geenrated before
                #we assume that simInput - the discount curve and the cpn curve follows their own stochastic paths
            elif model =='GBM':
                riskFactorModelling = RiskFactorModelling(vol, dt)
                simInput[row][column] = riskFactorModelling.geometricBrownian(simInput[row-1][column], resetVol)
                cpn_index_grid[row][column] = riskFactorModelling.geometricBrownian(cpn_index_grid[row-1][column], resetVol, rateCorrBM)
                variance_t = variance[row][column] = riskFactorModelling.hestonVol() if stochasticVol else variance_t
                vol = np.sqrt(variance_t)  # calculation spot vol is for the purpose of next Risk Factor object construction
            elif model == 'vasicek' :
                riskFactorModelling = RiskFactorModelling(vol, dt)
                simInput[row][column] = riskFactorModelling.vasicek(simInput[row -1][column],resetVol)
                cpn_index_grid[row][column] = riskFactorModelling.vasicek(cpn_index_grid[row -1][column],resetVol, rateCorrBM)
                variance_t = variance[row][column] = riskFactorModelling.hestonVol() if stochasticVol else variance_t
                vol= np.sqrt(variance_t)
            pv_grid[row][column], cashflowNb, pv_grid_ftp[row][column] = payoff(row, column,cashflowNb, simInput, cpn_index_grid, T)
    return simInput, pv_grid, cpn_index_grid, variance, pv_grid_ftp


# In[7]:


import time
start = time.time()
# start check time
tradeMaturityList = 0.5 # it could be a list
maturity = 3.0
swapRateSpread = 0.02 # this is the spread addon top of the simulated rates to constuitue a cpn
fixedLegCpn = 0.04 #0.041 # this is the pay leg, for DVA
percentile = 97.5 # [0,1009 nth percentil
notional = 1000000
CptyCollateral = 10000 # this is the counterparty posting collateral
sigma = 0.2 # this is the vol, not the variance
simStep = 50
riskFree_ir = 0.02 # here we try to seperate risk free rate and the cpn rate so that it is two different rates and follow its own dynamic process
cpn_index = 0.02  # cpn is where things have to change, which is a dynamics process
dcf = 0.5 #this is the date count faction
#df = 1 # T is in years # this needs to be changed to reflect the real time value  of rate
variance = []
stochasticVol = True # this flag determine whether or not we should use the the stochastic vol
rateCorrBM = 1.0
cptyspread = 0.018  # this should change to S_i(T_i-T_i-1) since here T_i-T_i-1) = 1 day
ownCDS = 0.0050
# the main issue of calculating the own PD as a constant is it ignore th eitme ffect which is the essence of this integration
#  ownPD = 1- np.exp(-ownCDS*tradeMaturity)
FTPSpread = ownCDS  # currently we assume the spread between FTP curve and OIS curve is the own CDS spread
#FTPSpread = 0.0050
model = 'vasicek'  # cir model doesn work well
resetVol = False
rateCorrBM = -1.0
# this is the flat to determine if we would like to have different vols for the two rates on each simulation path 3


# In[8]:


#for maturity in tradeMaturityList: # here the maturity is a list and it is for the purpose to compare
timeStep = int(maturity * 250)  # thsi means that it is daily dt, we assume annual business day is 250
timeStepList = range(0, timeStep)
PaymentTime = (1 / dcf) * maturity
dt = maturity / float(timeStep) # dt is daily
output = simulation(simStep, timeStep, sigma, dt, riskFree_ir, cpn_index,PaymentTime, stochasticVol, model)
#output=simulation()
EE = output[1] # this is the future MTM discounting by risk free rate
EPE = np.where(EE<0,0, EE)
ENE= np.where(EE>0,0, EE)
simulatedVol = np.sqrt(output[3])
simulatedCpn = output[2]
simulatedRFrate = output[0]
npv_ftp = output[4]
#print " my little stupid fva is", 0.04* dcf* notional*(0.987577800494), "substract", 0.04* dcf*notional*(0.990049833749)
end = time.time()
print ("time spent"), (end - start)


# In[9]:


meanEPE = [np.mean(EPE[i]) for i in range(0, timeStep)]  # expected positive exposure is for CVA calculation
meanENE = [np.mean(ENE[i]) for i in range(0, timeStep)]
meanEE = [np.mean(EE[i]) for i in range(0, timeStep)]
quantileEPE =[np.percentile(EPE[i], percentile) for i in range(0, timeStep)] # output[1] here is the array matrix shape of EPE
meanNPV_ftp = [np.mean(npv_ftp[i]) for i in range(0, timeStep)]
meanSimulatedRFrate = [np.mean(simulatedRFrate[i]) for i in range(0,timeStep)]
meanSimulatedCpn = [np.mean(simulatedCpn[i]) for i in range(0, timeStep)]
meanVol = [np.mean(simulatedVol[i]) for i in range(0, timeStep)]


# In[10]:




import matplotlib
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[11]:


#fig=plt.figure(figsize=(8, 8), dpi= 80, facecolor='w', edgecolor='k')
for i in range(0, simStep):
    plt.plot(np.transpose(EE)[i]) 
plt.plot(meanEPE,'bs')
plt.title('Expected Exposure')
print ('Model in use is ', model,' and the mean EE is', np.mean(meanEE),'and the max EE is', np.max(meanEE))
plt.grid()
plt.show()


# In[ ]:




