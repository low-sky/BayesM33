import emcee
import numpy as np
import scipy.stats as ss

data = np.loadtxt('M33_sample.txt')
wco, wco_err, whi,whi_err, dustsd = data[:,0],data[:,1],data[:,2],data[:,3],data[:,4]

def logprob(p,wco,sigco,whi,sighi,dustsd):
    gdr,alphaCO,sigdust,Kdark = p[0],p[1],p[2],p[3]
    a = 0.0146/gdr
    b = alphaCO/gdr
    c = Kdark/gdr
    if np.any(p[0:2]<0):
        return -np.inf
    term1 = -0.5*(c+b*wco+a*whi-dustsd)**2/(sigdust**2+b**2*sigco**2+a**2*sighi**2)
    term2 = -0.5*np.log(((a**2/sigdust**2)+(1/sighi**2)))
    term3 = -0.5*np.log((sigdust**2*sigco**2*sighi**2))
    term4 = -0.5*np.log((sigdust**2+b**2*sigco**2+a**2*sighi**2)/(sigdust**2*sigco**2+a**2*sigco**2*sighi**2))
    prior = ss.beta.logpdf(gdr/400,2,2)
    print(np.sum(term1),np.sum(term2),np.sum(term3),np.sum(term4))

    return(np.sum(term1+term2+term3+term4)+prior)

ndim = 4
nwalkers = ndim*10
p0 = np.zeros((nwalkers,ndim))
p0[:,0] = np.random.randn(nwalkers)*20+150
p0[:,1] = np.random.randn(nwalkers)*1+8.0
p0[:,2] = np.random.randn(nwalkers)*dustsd.std()/10
p0[:,3] = (np.random.randn(nwalkers))


sampler = emcee.EnsembleSampler(nwalkers,ndim,logprob,args=[wco,wco_err,whi,whi_err,dustsd])
pos,prob,state = sampler.run_mcmc(p0,300)
