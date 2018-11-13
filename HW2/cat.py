import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from sympy.solvers import solve
from sympy import *
import math
w1_mu = 5
w1_varian = 2
w1_sigma = math.sqrt(w1_varian)
w1_prior = 0.5
w1_x = np.linspace(w1_mu - 3*w1_sigma, w1_mu + 3*w1_sigma, 100)
plt.plot(w1_x, w1_prior*norm.pdf(w1_x, w1_mu, w1_sigma), label="w1 class (Happy cat)")
w2_mu = 0
w2_varian = 2
w2_sigma = math.sqrt(w2_varian)
w2_prior = 0.5
w2_x = np.linspace(w2_mu - 3*w2_sigma, w2_mu + 3*w2_sigma, 100)
plt.plot(w2_x, w2_prior*norm.pdf(w2_x, w2_mu, w2_sigma), label="w2 class (Sad cat)")
x = Symbol('x')
dec_boundry = solve(w1_prior*(1/(w1_sigma*sqrt(2*pi)))*(exp((-1/2)*((x-w1_mu)/w1_sigma)**2))-w2_prior*(1/(w2_sigma*sqrt(2*pi)))*(exp((-1/2)*((x-w2_mu)/w2_sigma)**2)), x, rational=False)
print("decision boundry for equaled priors = ", dec_boundry[0])
w1_prior = 0.8
w2_prior = 0.2
dec_boundry = solve(w1_prior*(1/(w1_sigma*sqrt(2*pi)))*(exp((-1/2)*((x-w1_mu)/w1_sigma)**2))-w2_prior*(1/(w2_sigma*sqrt(2*pi)))*(exp((-1/2)*((x-w2_mu)/w2_sigma)**2)), x, rational=False)
print("decision boundry for prior=0.8 for happy cat = ", dec_boundry[0])
plt.savefig("T2")
plt.show()
print("changing the distribution of the second class...")
w2_mu = 0
w2_varian = 4
w2_sigma = math.sqrt(w2_varian)
w1_prior = 0.5
w2_prior = 0.5
dec_boundry = solve(w1_prior*(1/(w1_sigma*sqrt(2*pi)))*(exp((-1/2)*((x-w1_mu)/w1_sigma)**2))-w2_prior*(1/(w2_sigma*sqrt(2*pi)))*(exp((-1/2)*((x-w2_mu)/w2_sigma)**2)), x, rational=False)
print("decision boundry for the new distribution of Sad cat = ", dec_boundry[0])
w1_x = np.linspace(w1_mu - 3*w1_sigma, w1_mu + 3*w1_sigma, 100)
plt.plot(w1_x, w1_prior*norm.pdf(w1_x, w1_mu, w1_sigma), label="w1 class (Happy cat)")
w2_x = np.linspace(w2_mu - 3*w2_sigma, w2_mu + 3*w2_sigma, 100)
plt.plot(w2_x, w2_prior*norm.pdf(w2_x, w2_mu, w2_sigma), label="w2 class (Sad cat)")
plt.savefig("OT1")
plt.show()