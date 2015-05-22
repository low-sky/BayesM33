import emcee
import numpy as np
import scipy.stats as ss
from scipy.special import erf
data = np.loadtxt('M33_sample.txt')
wco, wco_err, whi,whi_err, dustsd = data[:,0],data[:,1],data[:,2],data[:,3],data[:,4]

def logprob_erf(p,wco,sigco,whi,sighi,dustsd):
    gdr,alphaCO,sigdust,Kdark = p[0],p[1],p[2],p[3]
#    gdr,alphaCO,sigdust = p[0],p[1],p[2]
    a = 1/gdr
    b = alphaCO/gdr
    c = Kdark/gdr
#    c = 0
    if np.any(p[0:2]<0):
        return -np.inf
    term1 = -0.5*(c+b*wco+a*whi-dustsd)**2/\
            (sigdust**2+b**2*sigco**2+a**2*sighi**2)
    term2 = np.log(1+erf(\
        (-b*(c+a*whi-dustsd)*sigco**2+\
         wco*(sigdust**2+a**2*sighi**2))/\
        (2**0.5*sigco**2*(sigdust**2+a**2*sighi**2)*
         (sigdust**2+b**2*sigco**2+a**2*sighi**2)/\
         (sigco**2*(sigdust**2+a**2*sighi**2)))**0.5
        ))
    term3 = -0.5*np.log((a**2/sigdust**2+1/sighi**2)*
                        (sigdust**2*sigco**2*sighi**2)*
                        (sigdust**2+b**2*sigco**2+a**2*sighi**2)/\
                        (sigco**2*(sigdust**2+a**2*sighi**2)))
    prior = ss.uniform.logpdf(gdr/400,0,1)
    return(np.sum(term1+term2+term3+prior))


def logprob(p,wco,sigco,whi,sighi,dustsd):
    gdr,alphaCO,sigdust,Kdark = p[0],p[1],p[2],p[3]
    a = 0.0146/gdr
    b = alphaCO/gdr
    c = 0
#    c = Kdark/gdr
    if np.any(p[0:2]<0):
        return -np.inf
    term1 = -0.5*(c+b*wco+a*whi-dustsd)**2/(sigdust**2+b**2*sigco**2+a**2*sighi**2)
    term2 = -0.5*np.log(((a**2/sigdust**2)+(1/sighi**2)))
    term3 = -0.5*np.log((sigdust**2*sigco**2*sighi**2))
    term4 = -0.5*np.log((sigdust**2+b**2*sigco**2+a**2*sighi**2)/(sigdust**2*sigco**2+a**2*sigco**2*sighi**2))
    prior = ss.unform.logpdf(gdr/400,0,1)
    print(np.sum(term1),np.sum(term2),np.sum(term3),np.sum(term4))

    return(np.sum(term1+term2+term3+term4)+prior)

ndim = 4
nwalkers = ndim*10
p0 = np.zeros((nwalkers,ndim))
p0[:,0] = ss.uniform.rvs(size=nwalkers)*100+100
p0[:,1] = ss.uniform.rvs(size=nwalkers)*2+4
p0[:,3] = ss.uniform.rvs(size=nwalkers)*10
p0[:,2] = dustsd.mean()*(0.3+np.random.randn(nwalkers)*0.05)


sampler = emcee.EnsembleSampler(nwalkers,ndim,logprob_erf,args=[wco,wco_err,whi,whi_err,dustsd])
pos,prob,state = sampler.run_mcmc(p0,300)
sampler.reset()
pos,prob,state = sampler.run_mcmc(p0,1000)
