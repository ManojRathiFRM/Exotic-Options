# quanto option

import numpy as np
from scipy.stats import norm , multivariate_t , multivariate_normal

def quanto_option(S, K, T, r_d, r_f, sigma_s, sigma_fx, rho, FX_rate):
    """
    Price a quanto call option using Black-Scholes adjustment
    S: Current asset price
    K: Strike price
    T: Time to maturity
    r_d: Domestic interest rate
    r_f: Foreign interest rate
    sigma_s: Asset volatility
    sigma_fx: FX rate volatility
    rho: Correlation between asset and FX rate
    FX_rate: Current exchange rate
    """
    # Adjusted drift
    r_adj = r_f - rho * sigma_s * sigma_fx
    
    # Standard Black-Scholes parameters
    d1 = (np.log(S/K) + (r_adj + 0.5*sigma_s**2)*T) / (sigma_s*np.sqrt(T))
    d2 = d1 - sigma_s*np.sqrt(T)
    
    # Option price in foreign currency
    price_foreign = (S * norm.cdf(d1) - K * np.exp(-r_adj*T) * norm.cdf(d2))
    
    # Convert to domestic currency
    price_domestic = FX_rate * price_foreign
    
    return price_domestic

# Example usage
S = 100      # Asset price
K = 100      # Strike
T = 1.0      # Time to maturity
r_d = 0.05   # Domestic rate
r_f = 0.02   # Foreign rate
sigma_s = 0.2  # Asset volatility
sigma_fx = 0.1 # FX volatility
rho = 0.3    # Correlation
FX_rate = 1.5 # Exchange rate

price = quanto_option(S, K, T, r_d, r_f, sigma_s, sigma_fx, rho, FX_rate)
print(f"Quanto call option price: {price:.4f}")


# Rainbow Option

def rainbow_option_mc(S1, S2, K, T, r, sigma1, sigma2, rho, n_paths=10000):
    """
    Price a rainbow option (two assets) using Monte Carlo
    """
    # Correlation matrix
    corr_matrix = np.array([[1, rho], [rho, 1]])
    L = np.linalg.cholesky(corr_matrix)
    
    # Generate correlated normal random variables
    Z = np.random.standard_normal((2, n_paths))
    W = np.dot(L, Z)
    
    # Calculate terminal asset prices
    S1T = S1 * np.exp((r - 0.5*sigma1**2)*T + sigma1*np.sqrt(T)*W[0])
    S2T = S2 * np.exp((r - 0.5*sigma2**2)*T + sigma2*np.sqrt(T)*W[1])
    
    # Payoff is maximum of the two assets minus strike
    payoffs = np.maximum(np.maximum(S1T, S2T) - K, 0)
    
    # Discount payoffs
    price = np.exp(-r*T) * np.mean(payoffs)
    
    return price

# Example usage
S1 = 120     # First asset price
S2 = 110     # Second asset price
K = 100      # Strike
T = 1.0      # Time to maturity
r = 0.05     # Risk-free rate
sigma1 = 0.15 # First asset volatility
sigma2 = 0.2 # Second asset volatility
rho = 0.5    # Correlation

price = rainbow_option_mc(S1, S2, K, T, r, sigma1, sigma2, rho)
print(f"Rainbow option price: {price:.4f}")

# Barrier Option 

def up_and_out_call(S, K, H, T, r, sigma, steps=260):
    """
    Price an up-and-out barrier call option using Monte Carlo simulation
    """
    dt = T/steps
    paths = 100000
    
    # Generate paths
    Z = np.random.standard_normal((paths, steps))
    S_path = np.zeros((paths, steps+1))
    S_path[:,0] = S
    
    # Simulate price paths
    for t in range(1, steps+1):
        S_path[:,t] = S_path[:,t-1] * np.exp((r - 0.5*sigma**2)*dt + 
                                            sigma*np.sqrt(dt)*Z[:,t-1])
    
    # Check for barrier hits and calculate payoff
    barrier_hit = np.any(S_path > H, axis=1)
    final_prices = S_path[:,-1]
    payoffs = np.maximum(final_prices - K, 0)
    payoffs[barrier_hit] = 0
    
    # Calculate option price
    price = np.exp(-r*T) * np.mean(payoffs)
    return price

# Example usage
S = 100 # Current price
K = 100  # Strike
H = 120  # Barrier
T = 1.0  # Time to maturity
r = 0.05 # Risk-free rate
sigma = 0.2 # Volatility

price = up_and_out_call(S, K, H, T, r, sigma)
print(f"Up-and-out call price: {price:.4f}")