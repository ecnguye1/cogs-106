import numpy as np

class Metropolis:

    # Constructor method: initialize Metropolis object with 2 input arguments
    def __init__(self, logTarget, initialState):
        self.logTarget = logTarget
        self.current = initialState
        self.samples = [initialState]

    # private method: returns True if proposal is accepted and False otherwise
    def _accept(self, proposal):
        logRatio = self.logTarget(proposal) - self.logTarget(self.current)
        acceptanceProb = min(1, np.exp(logRatio))
        if np.random.rand() < acceptanceProb: # proposed state is accepted and becomes the new state
            self.current = proposal
            self.samples.append(proposal)
            return True
        else: # current state is retained
            self.samples.append(self.current)
            return False
        
    # adaptation phase of Metropolis algorithm
    def adapt(self, blockLengths):
        nBlocks = len(blockLengths)
        currentBlock = 0
        currentRate = 0.0
        # Gaussian proposal distribution starts with state mu = 0 and sigma = 1
        self.mu = 0.0
        self.sigma = 1.0
        # runs blocks of iterations and adjusts value of mu to achieve acceptance rate of approx 0.4
        while currentBlock < nBlocks:
            acceptances = 0
            proposals = 0
            for i in range(blockLengths[currentBlock]):
                proposal = self.current + np.random.normal(self.mu, self.sigma)
                if self._accept(proposal):
                    acceptances += 1
                proposals += 1
            acceptanceRate = acceptances / proposals
            self.mu = self.mu * ((acceptanceRate / 0.4) ** 1.1)
            currentRate = (currentRate * currentBlock + acceptanceRate) / (currentBlock + 1)
            currentBlock += 1
        return self
    
    # generates n samples from the target distribution using Metropolis algorithm
    def sample(self, nSamples):
        for i in range(nSamples):
            proposal = self.current + np.random.normal(self.mu, self.sigma)
            self._accept(proposal)
        return self
    
    # returns dictionary containing mean and 95% credible interval of generated samples
    def summary(self):
        samples = np.array(self.samples)
        return {
            'mean': np.mean(samples),
            'std': np.std(samples),
            'c025': np.percentile(samples, 2.5),
            'c975': np.percentile(samples, 97.5),
        }