import pandas as pd
import numpy as np
from sklearn.manifold
import TSNE
import matplotlib.pyplot as plt
from matplotlib
import figure
from sklearn.cross_decomposition
import PLSCanonical, PLSRegression, CCA from sklearn.impute
import SimpleImputer
from sklearn.preprocessing
import StandardScaler

# Import LBC data
lbc_csv = pd.read_csv("./data/LBC_Olink.csv")

# Variable label groups
proteins = ["IL8",
    "VEGFA",
    "MCP.3",
    "CDCP1",
    "CD244",
    "IL7",
    "OPG",
    "LAP.TGF.beta.1",
    "uPA",
    "IL6",
    "MCP.1",
    "CXCL11",
    "AXIN1",
    "TRAIL",
    "CXCL9",
    "CST5",
    "OSM",
    "CXCL1",
    "CCL4",
    "CD6",
    "SCF",
    "IL18",
    "SLAMF1",
    "TGF.alpha",
    "MCP.4",
    "CCL11",
    "TNFSF14",
    "FGF.23",
    "FGF.5",
    "MMP.1",
    "LIF.R",
    "FGF.21",
    "CCL19", 
    "IL.15RA", 
    "IL.10RB", 
    "IL.18R1",
    "PD.L1",
    "Beta.NGF",
    "CXCL5",
    "TRANCE",
    "HGF",
    "IL.12B",
    "MMP.10",
    "IL10",
    "CCL23",
    "CD5",
    "CCL3",
    "Flt3L",
    "CXCL6",
    "CXCL10",
    "X4E.BP1",
    "SIRT2",
    "CCL28",
    "DNER",
    "EN.RAGE",
    "CD40",
    "FGF.19",
    "MCP.2",
    "CASP.8",
    "CCL25",
    "CX3CL1",
    "TNFRSF9",
    "NT.3",
    "TWEAK",
    "CCL20",
    "ST1A1",
    "STAMBP",
    "ADA",
    "TNFB",
    "CSF.1"
    ]

# set covariates
age = ["ageyears_w2"] 
sex = ["sex"] 
factors = ["ICVc_gm_mm3_w2"]

# Filling in missing values with column mean# Use
imp = SimpleImputer(missing_values = np.nan, strategy = 'mean') imp.fit(lbc_csv[proteins + age + sex])
lbc_proteins = imp.transform(lbc_csv[proteins + age + sex]) imp.fit(lbc_csv[factors])
lbc_factors = imp.transform(lbc_csv[factors])

# Scale using statistical scoring
scaler = StandardScaler() X = np.array(lbc_proteins) scaler.fit(X)
X = scaler.transform(X)
Y = np.array(lbc_factors) scaler.fit(Y)
Y = scaler.transform(Y) print(X.shape) print(Y.shape)

# Fit PLS model
pls2 = PLSRegression(n_components = 3) pls2.fit(X, Y)


# Plot PLS loadings(importance of variables on model)# Y - axis shows relative contribution of protein to model
loadings = pls2.x_loadings_[: -2] ind = np.arange(len(loadings)) plt.figure(figsize = (20, 10)) plt.bar(ind, loadings[: , 0]) plt.xticks(ind, proteins)

subset = lbc_csv[proteins] X = np.array(subset) X.shape

# Fit TSNE model
X_emb = TSNE(n_components = 2, perplexity = 1, learning_rate = 100, n_iter = 1000, ␣􏰀→init = 'pca')
    .fit_transform(X)
X_emb.shape

# Colour labels based on risk factors
c_smoke = lbc_csv['smokprev_w2'] c_smokenow = lbc_csv['smokcurr_w2'] c_al = lbc_csv['alcfreq_w2']
c_sex = lbc_csv['sex']
c_gout = lbc_csv['gout_w1']
c_age = lbc_csv['agedays_w2']
c_bmi = lbc_csv['bmi_w2']
c_diab = lbc_csv['diab_w2']
c_chol = lbc_csv['hichol_w2']
c_six = lbc_csv['sixmwk_w2']
c_cog = lbc_csv['g']

c_all = (1 + c_smoke * 2) + (2 + c_al * 3)

diab = lbc_csv['diab']
cvdhist = lbc_csv['cvdhist'] * 10
hichol = lbc_csv['hichol'] * 20
stroke = lbc_csv['stroke'] * 30
parkin = lbc_csv['parkin'] * 40
hibp = lbc_csv['hibp'] * 50
demente = lbc_csv['demente'] * 60
code = diab + cvdhist + hichol + stroke + parkin + hibp + demente risk = code > 0

#Set brain metric predictors
gmr = lbc_csv['gmIcv_ratio_w2']
ratio = lbc_csv['brainIcv_ratio_w2']
gmv = lbc_csv['ICVc_gm_mm3_w2']

#Plot TSNE scatter plot
plt.figure(figsize = (10, 10)) 
plt.scatter(X_emb[: , 0], X_emb[: , 1], c = c_cog)
