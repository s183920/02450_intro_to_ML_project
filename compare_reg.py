import numpy as np
from scipy import stats
from __init__ import mcnemar


loss_ann = np.array([ 0.381250,0.381757 , 0.380246,0.381491,0.384566 ,0.380636,0.382878 ,0.383862 ,0.383854, 0.384667 ])
loss_linreg = np.array([0.762323 , 0.764036, 0.762564, 0.764180 , 0.765574, 0.762503, 0.761503, 0.761765, 0.763895, 0.763392 ])
loss_baseline = np.array([1.015791, 0.989860, 1.001790, 0.995425, 0.979062, 1.005317, 1.017560, 1.010184,  0.991158,  0.993859])

loss_KNN = np.array([0.017, 0.021,0.023,0.023,0.024,0.025,0.028,0.029,0.031,0.034])
loss_logreg = np.array([0.02,0.0,0.0,0.0,0.1,0.1,0.0,0.0,0.1,0.1])
loss_baseline = np.array([0.1312, 0.2382, 0.2412, 0.0776, 0.979062, 0.00705, 0.3433, 0.0239,  0.2245,  0.1989])
loss_KNN = np.array([0.017,0.021,0.023,0.023,0.024,0.025,0.028,0.029,0.031,0.034])
loss_logreg = np.array([0.02,0.0,0.0,0.0,0.01,0.01,0.0,0.0,0.01,0.0])
loss_baseline2 = np.array([0
.281,0.1312,0.2382,0.2412,0.0776,0.00705,0.3433,0.0239,0.2245,0.1989])

def reg_ttest(loss1, loss2, alhpa = 0.05):
    """
    loss1 = loss of model 1
    loss2 = loss of model 2
    """
    z = loss1 - loss2

    zhat = z.mean()
    n = len(z)
    shat = 1/(n*(n-1)) * np.sum((z-zhat)**2)

    s = stats.ttest_rel(loss1, loss2)

    return s.pvalue
    
print("P-value = {}".format(reg_ttest(loss_ann, loss_linreg)))
print("P-value = {}".format(reg_ttest(loss_ann, loss_baseline)))
print("P-value = {}".format(reg_ttest(loss_baseline, loss_linreg)))

print("P-value = {}".format(reg_ttest(loss_KNN, loss_logreg)))
print("P-value = {}".format(reg_ttest(loss_KNN, loss_baseline)))
print("P-value = {}".format(reg_ttest(loss_baseline, loss_logreg)))
print("P-value = {}".format(reg_ttest(loss_KNN, loss_baseline2)))
print("P-value = {}".format(reg_ttest(loss_baseline2, loss_logreg)))


