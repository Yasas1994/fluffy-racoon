import pandas as pd
import numpy as np
import sklearn as sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import  auc, f1_score,roc_curve, roc_auc_score, recall_score, precision_score
sns.set()


parser = argparse.ArgumentParser(description='Plots ROC given scores and true lables')
parser.add_argument('-outfile',help='output png file name')
parser.add_argument('-infile', help='path to input csv file')

args = parser.parse_args()

print(args)

data = pd.read_csv(args.infile)

colormap='coolwarm' #change color theme 
SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 28

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title




# Compute ROC curve and ROC area for each class
lables= data['lables']#binary lables 
scores= data['score']#scores between 0-1

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(2):
    fpr[i], tpr[i], thresholds = roc_curve(lables, scores)
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(lables, scores)
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

 #creating the roc curve

plt.figure(figsize=(10,10),)
ax = plt.subplot()
#plt.rcParams.update({'font.size': 34})
lw = 2
xx=ax.scatter(
    fpr[1],
    tpr[1],
    c=cm.coolwarm(np.abs(thresholds)),
    cmap='Blues',
    lw=lw,
    label="ROC curve (area = %0.2f)" % roc_auc[1],
)
xx.set_cmap(colormap)

plt.xlim([-0.010, 1.0])
plt.ylim([-0.010, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic")
plt.legend(loc="lower right")
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

plt.colorbar(xx, cax=cax)
cax.set_ylabel("Threshold")
#plt.scatter(x,y, c=cm.hot(np.abs(y)), edgecolor='none')
ax.plot([0, 1], [0, 1], color='grey', lw=lw, linestyle="--")

plt.savefig(args.outfile, bbox_inches='tight')
