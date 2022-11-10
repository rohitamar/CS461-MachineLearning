#imports
import math
import numpy as np

#constants
size = 30
prob = 0.55

samples = np.random.binomial(n = 1, p = prob, size = size) #get 30 bernoulli trials
sample_mean = samples.mean() #find their average to find p_hat
print('p_hat: ', sample_mean) #p_hat = sample_mean

#cdf function
#accumulates probabilities from i = 501 to 1000 (see pdf)
#essentially, this function finds the probability of candidate A winning the election
def cdf(p):
    pb = 0
    for i in range(501, 1001):
        pb += math.comb(1000, i) * pow(p, i) * pow(1 - p, 1000 - i)
    return pb

#probability that candidate A wins with p = 0.55 
print('Probability of Candidate A winning on p: ', cdf(0.55))

mu = sample_mean
sigma = math.sqrt((mu * (1 - mu)) / size) #variance is p_hat*(1-p_hat) / k and numpy takes in standard deviation in the normal function

avg, i = 0, 0 #variables used in the CLT sampling (question 1.7)
while i < 1000: #1000 samples
    clt = np.random.normal(mu, sigma, 1) #get probability from normal distribution
    if clt[0] >= 0 and clt[0] <= 1: #check if it's >= 0 and <= 1
        avg += cdf(clt[0]) #find the probability that candidate A wins with this probability
        i += 1 #increment trial
avg = avg / 1000 #find average

#print
print('Probability of Candidate A winning on CLT sampled p_hat: ', avg)
