import numpy as np

# parameters for J evolution
a_coefr = np.array([0.43739546, 1.31628209])
b_coef,b_index,b_coefr = np.array([0.58066774, 0.58081087, 1.29275231])
c_coef,c_index = np.array([0.72997304, 0.20851588])
d_coef = 0.8733596630878311
d_index = 0.4972446491935101

# parameters for rmax evolution
av_coefs = np.array([0.37,12.,1.21])
bv_coef,bv_index,bv_coefr = np.array([0.28, 0.58, 1.36])
cv_coef,cv_index = np.array([0.78, 0.22])

# parameters for vmax evolution
ar_coefs = np.array([0.53,84.,1.26])
br_coef,br_index,br_coefr = np.array([0.48, 0.38, 0.70])
cr_coef,cr_index = np.array([0.91, 0.13])