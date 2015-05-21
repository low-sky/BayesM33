from astropy.table import Table
import scipy.stats as ss
import numpy as np
import emcee
import matplotlib.pyplot as plt
#import seaborn as sns
#import pandas as pd

# Set up the log-probability function

def lp(p,Ico,errIco,SigmaHI,errSigmaHI,SigmaDust):
    gdr, alphaCO, c, errSigmaDust = p[0], p[1], p[2], p[3]

    if np.any(p<0):
        return(-np.inf)
#    if errSigmaDust < 0:
#        return(-np.inf)
    
    y = Ico / (np.sqrt(2)*errIco)
    t = SigmaHI / (np.sqrt(2)*errSigmaHI)
    Gamma = (gdr*SigmaDust - c)/(np.sqrt(2)*errSigmaDust)
    alpha = alphaCO*errIco/(errSigmaDust)
    beta = errSigmaHI/(errSigmaDust)
    
    prefactor =-0.5*np.sum(np.log((1+alpha**2+beta**2)*errSigmaDust**2))
#    term1 = (y**2+Gamma**2+beta**2*y**2-2*Gamma*beta*t)*\
#            (1+alpha**2+beta**2)
#    term2 = (y+Gamma*alpha+beta**2*y-alpha*beta*t)**2

#    term3 = (Gamma**2+alpha**2*y**2+alpha**2*t**2+t**2-2*Gamma*y*alpha)*\
#            (1+alpha**2+beta**2)
#    term4 = (t+Gamma*beta+alpha**2*t-alpha*beta*y)**2


    dataterm = -np.sum((Gamma-alpha*y-beta*t)**2/(1+alpha**2+beta**2))

#    print(np.sum(term1-term2),np.sum(term3-term4))


#    dataterm = -(term3-term4)/((1+beta**2)*(1+alpha**2+beta**2))
    
    lp = (dataterm)+prefactor

    if np.isnan(lp):
        return(-np.inf)
#    print(lp,p)
    return(lp)

t = Table.read('M33_sample.txt',format='ascii.commented_header')
#plt.scatter(t['Sigma_HI'],t['WCO10'])
#plt.show()



ndim, nwalkers = 4,50
p0 = np.zeros((nwalkers,ndim))

p0[:,0] = ss.uniform.rvs(size=nwalkers)*100+100
p0[:,1] = ss.uniform.rvs(size=nwalkers)*2+4
p0[:,2] = ss.uniform.rvs(size=nwalkers)*10
p0[:,3] = np.median(t['Sigma_Dust'])*(0.3+0.05*ss.norm.rvs(size=nwalkers))


sampler = emcee.EnsembleSampler(nwalkers, ndim,
                                lp,
                                args=[t['WCO10'],t['err_WCO10'],
                                      t['Sigma_HI'],t['err_Sigma_HI'],
                                      t['Sigma_Dust']])

pos, prob, state = sampler.run_mcmc(p0, 400)
sampler.reset()
sampler.run_mcmc(pos,1000)

# plotting

frames = pd.DataFrame(sampler.flatchain,columns=['GDR','alphaCO','c','sigdust'])
g = sns.JointGrid("GDR","alphaCO",frames,space=0)
g.plot_marginals(sns.kdeplot, shade=True)
g.plot_joint(sns.kdeplot, shade=True, cmap="PuBu", n_levels=40)
plt.savefig('marginals.pdf')