# Importing necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set max row to 300
pd.set_option('display.max_rows', 300)

class MonteCarloOptionPricing:
    ''' Monte Carlo Option Pricing Engine'''
    #simulating price path for the year with each time step of one day
    def __init__(self, S0:float, strike:float, rate:float, sigma:float, dte:int,nsim:int, timesteps:int=252) -> float:
        
        self.S0 = S0 # Initial stock price
        self.K = strike # Strike price
        self.r = rate # Risk-free interest rate
        self.sigma = sigma # Volatility
        self.T = dte # Time to expiration
        self.N = nsim # Number of simulations
        self.ts = timesteps # Time steps
        self.df = np.exp(-self.r * self.T) # discount factor

    @property #to simple call the function
    def psuedorandomnumber(self):
        ''' generate psuedo random numbers'''
        return np.random.standard_normal(self.N)

    @property
    def simulatepath(self):
        ''' simulate price path'''
        np.random.seed(50000) #Number of simulations
        
        # define dt
        dt = self.T/self.ts #1/252
        
        # simulate path
        S = np.zeros((self.ts, self.N))  #simulation in column for numpy vectorization
        S[0] = self.S0
        
        for i in range(0, self.ts-1):
            w = np.random.standard_normal(self.N)
            S[i+1] = S[i] * (1+ self.r*dt + self.sigma*np.sqrt(dt)*w) #euler mariyama discretization of SDE using the formula defined above
        return S
        
      
    
    @property
    def asianoptionfixed(self):
            ''' calculate asian option fixed strike payoff''' #keeping the strike fixed
            S = self.simulatepath
        
            # average the price across days
            A = S.mean(axis=0) #across the rows taking  Arithmetic average
            # calculate the discounted value of the expected payoff
            asian_fixedcall = self.df * np.mean(np.maximum(0, A-self.K))
            asian_fixedput = self.df * np.mean(np.maximum(0, self.K-A))
            return [asian_fixedcall, asian_fixedput]

    @property
    def asianoptionfloating(self):
            ''' calculate asian option floating payoff''' #keeping the strike floating
            S = self.simulatepath
        
            # average the price across days
            A = S.mean(axis=0) #across the rows taking  Arithmetic average
            # calculate the discounted value of the expected payoff
            asian_floatingcall = self.df * np.mean(np.maximum(0, A- S[-1]))
            asian_floatingput = self.df * np.mean(np.maximum(0, S[-1]-A))
            return [asian_floatingcall, asian_floatingput]

    @property
    def lookbackoptionfixed(self):
            ''' calculate lookback option fixed strike payoff''' #keeping the strike fixed and realised maximum
            S = self.simulatepath
            # maximum and minimum price across days
            lookback_callpayoffs = np.max(S, axis=0) - self.K  # Lookback call option payoffs
            lookback_putpayoffs = self.K - np.min(S, axis=0)  # Lookback put option payoffs

            lookback_fixedcall = self.df * np.mean(np.maximum(lookback_callpayoffs, 0))  # Discounted expected payoff for lookback call option
            lookback_fixedput = self.df * np.mean(np.maximum(lookback_putpayoffs, 0))  # Discounted expected payoff for lookback put option
            return [lookback_fixedcall,lookback_fixedput]


    @property
    def lookbackoptionfloating(self):
            ''' calculate lookback option floating strike payoff''' #keeping the strike floating and realised maximum
            S = self.simulatepath
            # maximum and minimum price across days
            lookback_callpayoffs = np.max(S, axis=0) - S[-1]  # Lookback call option payoffs
            lookback_putpayoffs = S[-1] - np.min(S, axis=0)  # Lookback put option payoffs

            lookback_floatingcall = self.df * np.mean(np.maximum(lookback_callpayoffs, 0))  # Discounted expected payoff for lookback call option
            lookback_floatingput = self.df * np.mean(np.maximum(lookback_putpayoffs, 0))  # Discounted expected payoff for lookback put option
            return [lookback_floatingcall,lookback_floatingput]

# Create a DataFrame
data = {'Scenario Name': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'],
        'Stock Price': [100, 110, 120, 100, 100, 100, 100, 100, 100],
        'Strike': [100, 100, 100, 110, 120, 100, 100, 100, 100],
        'Time to Expiry (in year)': [1, 1, 1, 1, 1, 1, 1, 1, 1],
        'Volatility': [0.20, 0.20, 0.20, 0.20, 0.20, 0.25, 0.15, 0.20, 0.20],
        'Risk Free Rate': [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.07, 0.03]}

df = pd.DataFrame(data)

# Print the DataFrame
print(df)

# instantiate
mc = MonteCarloOptionPricing(100,100,0.05,0.2,1,100000) #inputting initial parameters
# Verify the generated price paths
pd.DataFrame(mc.simulatepath).head(2)

mc.simulatepath[-1]

# Plot the histogram
plt.hist(mc.psuedorandomnumber, bins=100);

#Plot the histogram of the simulated price path at maturity
plt.hist(mc.simulatepath[-1], bins=100);

# Plot initial 200 simulated path using matplotlib
plt.plot(mc.simulatepath[:,:200])
plt.xlabel('time steps')
plt.xlim(0,252)
plt.ylabel('index levels')
plt.title('Monte Carlo Simulated Asset Prices');

# Get option values
print(f"Asian Call Option Value is {mc.asianoptionfixed[0]:0.4f}") 
print(f"Asian Put Option Value is {mc.asianoptionfixed[1]:0.4f}")

# Get option values
print(f"Asian Call Option Value is {mc.asianoptionfloating[0]:0.4f}")
print(f"Asian Put Option Value is {mc.asianoptionfloating[1]:0.4f}")

# Get option values
print(f"Lookback Call Option Value is {mc.lookbackoptionfixed[0]:0.4f}")
print(f"Lookback Put Option Value is {mc.lookbackoptionfixed[1]:0.4f}")

# Get option values
print(f"Lookback Call Option Value is {mc.lookbackoptionfloating[0]:0.4f}")
print(f"Lookback Put Option Value is {mc.lookbackoptionfloating[1]:0.4f}")








  
