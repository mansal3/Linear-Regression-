import pandas as pd
from sklearn.model_selection import train_test_split

dataset=pd.read_csv("C:/Users/Manpreet.saluja/Downloads/50_Startups.csv")

dataset.head()
dataset.tail()

import seaborn as sns
sns.pairplot(dataset)

dataset.corr()

import statsmodels.formula.api as sum

new_data = dataset.rename(columns = {"R&D Spend": "RD", 
                                  "Administration":"Admin", 
                                  "Marketing Spend": "Marketing",
                                  "Profit":"P"}) 

#modeling
model=sum.ols('P~RD+Admin+Marketing',data=new_data).fit()

#getting coefficient B0+b1+b2+b3
model.params

#summary
model.summary()
# in summary we found Admin And Marketing as to e insignficant hence we will check and build model with them to check weather they significant

#preparing model with Admin
model_admin=sum.ols('P~Admin',data=new_data).fit()
model_admin.summary()
#insignificant

model_marketing=sum.ols('P~Marketing',data=new_data).fit()
model_marketing.summary()
#significant

model_marketing_admin=sum.ols('P~Marketing+Admin',data=new_data).fit()
model_marketing_admin.summary()
#significant

import statsmodels.api as ss
ss.graphics.influence_plot(model)

#record 49 is influence record ->act lke a outlier

dataset_new=new_data.drop(new_data.index[[49]])

#againing building the model

model_new_=sum.ols('P~RD+Admin+Marketing',data=dataset_new).fit()
model.summary()
#still insignificant
#hence continue with variables
#now we are left with nothing else deleting columns
#now how to decide which column to delte that should be decide by VIF

res_admin=sum.ols('Admin~Marketing+RD',data=new_data).fit().rsquared
vif_admin=1/(1-res_admin)

res_marketing=sum.ols('Marketing~Admin+RD',data=new_data).fit().rsquared
vif_marketing=1/(1-res_marketing)

res_RD=sum.ols('RD~Marketing+Admin',data=new_data).fit().rsquared
vif_RD=1/(1-res_RD)

d1={'Columns':pd.Series(['RD','Marketing','Admin']),'VIF':pd.Series([vif_RD,vif_admin,vif_marketing])}

vif_frame=pd.DataFrame(d1,columns=['Columns','VIF'])

ss.graphics.plot_partregress_grid(model)
#admin is least affecting any thing


model_new_1=sum.ols('P~RD+Marketing',data=dataset_new).fit()
model_new_1.summary()

#predicting

pred=model_new_1.predict(dataset_new[['Admin','Marketing','RD']])

#residual
resid=pd.DataFrame(pd.Series(dataset_new['P']-pred))
resid

resid.mean()

resid.rename(columns={'P':'residual'},inplace='true')

from matplotlib import pyplot as plt
plt.scatter(x=pred,y=resid,color='red')