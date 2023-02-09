import numpy as np
from scipy.stats import norm

class SignalDetection:
    
    def __init__(self, hits, misses, falseAlarms, correctRejections):
        self.hits = hits
        self.misses = misses
        self.fa = falseAlarms
        self.cr = correctRejections
        #calculating H-hit rate
        self.h_rate = self.hits / (self.hits + self.misses)
        #calculating FA false alarm rate
        self.f_rate = self.fa / (self.fa + self.cr)
        
    def d_prime(self):
        z_h = norm.ppf(self.h_rate)
        z_fa = norm.ppf(self.f_rate)
        d_p = z_h - z_fa
        return d_p
    
    def criterion(self):
        z_h = norm.ppf(self.h_rate)
        z_fa = norm.ppf(self.f_rate)
        #calculating base on C = -0.5 * (Z(H) + Z(FA))
        criterion = -0.5 * (z_h  + z_fa)
        return criterion
