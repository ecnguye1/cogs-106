import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

class SignalDetection:
    def __init__(self, hits, misses, falseAlarms, correctRejections):
        self.__hits = hits 
        self.__misses = misses
        self.__falseAlarms = falseAlarms 
        self.__correctRejections = correctRejections

    def hit_rate(self):
        return self.__hits / (self.__hits + self.__misses)

    def falseAlarm_rate(self):
        return self.__falseAlarms / (self.__falseAlarms + self.__correctRejections)

    def d_prime(self):
        hit_rate = self.hit_rate()
        falseAlarm_rate = self.falseAlarm_rate()
        z_hit = stats.norm.ppf(hit_rate) 
        z_falseAlarm = stats.norm.ppf(falseAlarm_rate)
        return z_hit-z_falseAlarm

    def criterion(self):
        hit_rate = self.hit_rate()
        falseAlarm_rate = self.falseAlarm_rate()
        z_hit = stats.norm.ppf(hit_rate)
        z_falseAlarm = stats.norm.ppf(falseAlarm_rate)
        return -0.5*(z_hit + z_falseAlarm)

    def __add__(self, other):
        return SignalDetection(self.__hits + other.__hits, self.__misses + other.__misses, 
                               self.__falseAlarms + other.__falseAlarms, 
                               self.__correctRejections + other.__correctRejections)
    
    def __mul__(self, scalar):
        return SignalDetection(self.__hits * scalar, self.__misses * scalar, 
                               self.__falseAlarms * scalar, 
                               self.__correctRejections * scalar)

    def plot_roc_old(self):
        hit_rate = self.hit_rate()
        falseAlarm_rate = self.falseAlarm_rate()
        plt.plot([0, falseAlarm_rate, 1], [0, hit_rate, 1], 'b')
        plt.scatter(falseAlarm_rate, hit_rate, c='r')
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.xlabel('False Alarm rate')
        plt.ylabel('Hit rate')
        plt.title('Receiver Operating Characteristic (ROC) curve')
        plt.show()

    def plot_sdt(self):
        hit_mu = 0
        hit_sigma = 1
        fa_mu = self.d_prime()
        fa_sigma = 1
        x = np.linspace(-4, 4, 500)
        hit_pdf = stats.norm.pdf(x, hit_mu, hit_sigma)
        fa_pdf = stats.norm.pdf(x, fa_mu, fa_sigma) 
        fig, ax = plt.subplots()
        ax.plot(x, hit_pdf, 'r', label='Signal')
        ax.plot(x, fa_pdf, 'g', label='Noise')
        ax.axvline((self.d_prime()/2)+self.criterion(), linestyle='--', color='b', label = "criterion")
        ax.hlines(max(hit_pdf), self.d_prime(), 0, colors='k', linestyles='dashed', label = "d-prime")
        ax.set_xlabel('Evidence')
        ax.set_ylabel('Probability Density')
        ax.set_title('Signal Detection Theory (SDT) Plot')
        ax.legend()
        plt.show()

#_______________________________________________________________________________________________________
# Assignment 4: Simulate and recover

    # creates one or more SignalDetection objects with simulated data
    @staticmethod
    def simulate(dprime, criteriaList, signalCount, noiseCount):
        sdtList = []
        for i in range(len(criteriaList)):
            criteria = criteriaList[i]
            K = criteria + dprime/2
            hit = [1 - norm.cdf(K - dprime)]
            fA = [1 - norm.cdf(K)]
            hits = np.random.binomial(n = signalCount, p = hit)
            misses = signalCount - hits
            falseAlarms = np.random.binomial(n = noiseCount, p = fA)
            correctRejections = noiseCount - falseAlarms
            sdt = SignalDetection(hits, misses, falseAlarms, correctRejections)
            sdt.hits = hits
            sdt.misses = misses
            sdt.falseAlarms = falseAlarms
            sdt.correctRejections = correctRejections
            sdtList.append(sdt)
        return sdtList
    
    # takes multiple SignalDetection objects and plots ROC curve
    @staticmethod
    def plot_roc(sdtList):
        plt.figure()
        plt.plot([0,1], [0,1], 'k--')
        plt.xlabel('false alarm rate')
        plt.ylabel('hit rate')
        plt.title('ROC curve')
        for sdt in sdtList:
            hit_rate = sdt.hit_rate()
            falseAlarm_rate = sdt.falseAlarm_rate()
            plt.plot(falseAlarm_rate, hit_rate, 'ko')
        plt.show()

    # calculates the negative log-likelihood of a SignalDetection object given an arbitrary hit rate and false alarm rate
    def nLogLikelihood(self, hit_rate, falseAlarm_rate):
        miss_rate = 1 - hit_rate
        correctRejection_rate = 1 - falseAlarm_rate
        likelihood = ((hit_rate ** self.__hits) * (miss_rate ** self.__misses) * 
                      (falseAlarm_rate ** self.__falseAlarms) * 
                      (correctRejection_rate ** self.__correctRejections))
        ell = -np.log(likelihood)
        return ell
    
    # compute a one-parameter ROC curve function
    @staticmethod
    def rocCurve(falseAlarm_rate, a):
        return norm.cdf(a + norm.ppf(falseAlarm_rate))

    # fits the one-parameter function to observed hit rate, false alarm rate pairs
    @staticmethod
    def fit_roc(sdtList):
        hit_rates = []
        falseAlarm_rates = []
        plt.plot([0,1], [0,1], 'k--')
        for sdt in sdtList:
            hit_rate = sdt.hit_rate()
            falseAlarm_rate = sdt.falseAlarm_rate()
            hit_rates.append(hit_rate)
            falseAlarm_rates.append(falseAlarm_rate)
            plt.plot(falseAlarm_rate, hit_rate, 'ko')
        # fitting the function: finding value of 'a' that minimizes loss
        def loss(a):
            sumOfSquares = 0
            for i in range(len(hit_rates)):
                p_rate = SignalDetection.rocCurve(falseAlarm_rates[i], a)
                sumOfSquares += (p_rate - hit_rates[i])**2
            return sumOfSquares
        result = minimize(loss, [0])
        a = result.x[0]
        # plot the ROC curve with the fitted curve
        x = np.linspace(0, 1, num=100)
        y = SignalDetection.rocCurve(x, a)
        plt.plot(x, y, 'r-')
        plt.xlabel('false alarm rate')
        plt.ylabel('hit rate')
        plt.title('ROC curve')
        plt.show()
        return a

    # evaluates the loss function L(a)
    @staticmethod
    def rocLoss(a, sdtList):
        loss = 0.0
        for sdt in sdtList:
            falseAlarm_rate = sdt.falseAlarm_rate() # observed false alarm rate
            hit_rate = SignalDetection.rocCurve(falseAlarm_rate, a) # predicted hit rate
            loss += sdt.nLogLikelihood(hit_rate, falseAlarm_rate)
        return loss
    
#_______________________________________________________________________________________________________
# Assignment 3 plots

# sd = SignalDetection(10, 30, 10, 5)
# sd.plot_roc_old()
# sd.plot_sdt()

#_______________________________________________________________________________________________________
# Assignment 4 plots

sdtList = []
sdtList.append(SignalDetection(11, 1, 15, 5))
sdtList.append(SignalDetection(14, 1, 7, 5))
sdtList.append(SignalDetection(11, 5, 5, 10))
sdtList.append(SignalDetection(8, 5, 1, 5))
sdtList.append(SignalDetection(17, 3, 10, 5))

SignalDetection.plot_roc(sdtList)
SignalDetection.fit_roc(sdtList)