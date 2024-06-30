#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats  #for the statistical tests
from scipy import stats as stats
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


loan=pd.read_csv('LoansData.csv')


# In[3]:


loan.head(2)


# (a). Intrest rate is varied for different loan amounts (Less intrest charged for high loan amounts)

# In[5]:


loan.dtypes


# In[6]:


loan.shape


# In[7]:


loan['Interest.Rate']=loan['Interest.Rate'].str.replace("%"," ")


# In[8]:


loan['Interest.Rate']=loan['Interest.Rate'].astype("float")


# In[9]:


## missing value treatment


# In[10]:


loan['Interest.Rate'].isnull().sum()


# In[11]:


loan['Interest.Rate'].fillna(loan['Interest.Rate'].mean(),inplace=True)


# In[12]:


loan['Interest.Rate'].isnull().sum()


# In[13]:


loan['Amount.Requested'].isnull().sum()


# In[14]:


loan['Amount.Requested'].fillna(loan['Amount.Requested'].mean(),inplace=True)


# In[15]:


loan['Amount.Requested'].isnull().sum()


# In[16]:


# Ho=Intrest rate does not vary for  different loan amount(Less intrest is not charged for high loan amount).
#H1=Intrestt rate is  varied for different loan amount(Less intrest charged for high loan amounts).
# Taking Confidence Interval at 95%, and p-value 0.05
# Finding the  relation between these two continous variables using Pearson test
stats.pearsonr(loan['Interest.Rate'],loan['Amount.Requested'])


# In[17]:


# Hence p-value smaller than 0.05 that means with 95% confidence, I will reject the null hypothesis
# Conclusion 
#Intrest rate is varied for different loan amounts (Less intrest charged for high loan amounts).
print("Conclusion :")
print("Intrest rate is varied for different loan amounts (Less intrest charged for high loan amounts)")


# b. Loan length is directly effecting intrest rate.

# In[18]:


loan.head(5)


# In[19]:


loan.dtypes


# In[20]:


loan['Loan.Length'].nunique()


# In[21]:


loan_36 =loan.loc[loan['Loan.Length']=='36 months','Interest.Rate']
loan_60 =loan.loc[loan['Loan.Length']=='60 months','Interest.Rate']


# In[22]:


# Ho=Loan length is not effecting intrest rate.
# H1=Loan length is directly effecting intrest rate
# Taking Confidence Interval at 95%, and p-value 0.05
# performing statistical analysis using t-test
stats.ttest_ind(loan_36,loan_60)


# In[23]:


# Hence p-value smaller than 0.05 that means with 95% confidence, I will reject the null hypothesis
# Conclusion 
# Loan length is directly effecting intrest rate
print("Conclusion :")
print("Loan length is directly effecting intrest rate")


# (c). Inrest rate varies for different purpose of loans

# In[24]:


# Ho=Interest rate does not vary for different purpose of loans.
# H1=Interest rate varies for different purpose of loans
# Taking Confidence Interval at 95%, and p-value 0.05
stats.spearmanr(loan["Interest.Rate"],loan["Loan.Purpose"])


# In[25]:


# Hence p-value smaller than 0.05 that means with 95% confidence, I will reject the null hypothesis
# Conclusion 
# Interest rate varies for different purpose of loans
print("Conclusion :")
print("Interest rate varies for different purpose of loans")


# (d).There is relationship between FICO scores and Home Ownership. It means  that, People with owning home will have high FICO scores.

# In[26]:


loan.head(2)


# In[27]:


loan['FICO.Range'].isnull().sum()


# In[28]:


loan['FICO.Range'].describe()


# In[29]:


loan['FICO.Range'].fillna("670-674",inplace=True)


# In[30]:


# Ho=There is no relationship between FICO scores and Home Ownership
# H1=There is relationship between FICO scores and Home Ownership.
# Taking Confidence Interval at 95%, and p-value 0.05
relation_FICO_and_homeownership= pd.crosstab(loan["Home.Ownership"],loan["FICO.Range"])


# In[31]:


stats.chi2_contingency(relation_FICO_and_homeownership)


# In[32]:


# Hence p-value smaller than 0.05 that means with 95% confidence, I will reject the null hypothesis
# Conclusion 
# There is relationship between FICO scores and Home Ownership. It means that, People with owning home will have high FICO scores.
print("Conclusion :")
print("There is relationship between FICO scores and Home Ownership. It means that, People with owning home will have high FICO scores.")


# # Business Problem 2

# (a)We would like to assess if there is any difference in the average 
# price quotes provided by Mary and Barry.
# 

# In[33]:


price=pd.read_csv('Price_Quotes.csv')


# In[34]:


price


# In[35]:


price.describe()


# In[36]:


# Ho=There is no difference in the average price quotes provided by Mary and Barry
# H1=There is difference in the average price quotes provided by Mary and Barry
# Taking Confidence Interval at 95%, and p-value 0.05
# H1=There is relationship between FICO scores and Home Ownership.
# Taking Confidence Interval at 95%, and p-value 0.05
stats.ttest_rel(price["Mary_Price"],price["Barry_Price"])


# In[37]:


# Hence p-value smaller than 0.05 that means with 95% confidence, I will reject the null hypothesis
# Conclusion 
# There is difference between the average price quotes provided by Mary and Barryprint
print("Conclusion :")
print("There is difference between the average price quotes provided by Mary and Barry")


# # Business problem 3

# (a).  Determine what effect, if any, the reengineering effort had on the 
# incidence behavioral problems and staff turnover. i.e To determine if the reengineering effort
# changed the critical incidence rate. Isthere evidence that the critical incidence rate
# improved?

# In[38]:


treat=pd.read_csv('Treatment_Facility.csv')


# In[39]:


treat


# In[40]:


treat.dtypes


# In[41]:


treat.rename(columns={"VAR4":"TRFF", "VAR5": "CI"}, inplace =True )


# In[42]:


prior= treat[treat["Reengineer"]=="Prior"]["CI"]
post= treat[treat["Reengineer"]=="Post"]["CI"]


# In[43]:


# Ho=There is no effect post the reengineer
# H1=There is effect post the reengineer
# Taking Confidence Interval at 95%, and p-value 0.05
stats.ttest_ind(prior,post)


# In[44]:


# HenceI got p-value greater than 0.05 that means with 95% confidence, I will fail to reject the null hypothesis
# Conclusion 
# There is insufficient evidence to conclude that the reengineering effort resulted in a significant change in the critical incidence rate 
print("Conclusion :")
print("There is insufficient evidence to conclude that the reengineering effort resulted in a significant change in the critical incidence rate ")


# # Business Problem 4

#  We will focus on the prioritization system. If the system is working, then
# high priority jobs, on average, should be completed more quickly than medium priority jobs,
# and medium priority jobs should be completed more quickly than low priority jobs. Use the
# data provided to determine whether thisis, in fact, occurring.

# In[45]:


prio=pd.read_csv('Priority_Assessment.csv')


# In[46]:


prio


# In[47]:


prio.dtypes


# In[48]:


prio['Priority'].nunique()


# In[49]:


prio.describe()


# In[50]:


high= prio[prio["Priority"]=="High"].Days
medium= prio[prio["Priority"]=="Medium"].Days
low= prio[prio["Priority"]=="Low"].Days


# In[51]:


# Ho=The prioritization system is not working
# H1= The prioritization system is working
# Taking Confidence Interval at 95%, and p-value 0.05


# In[52]:


# as i have 3 groups with me, i will use ANOVA test - Analysis of Variance (basically helps in finding the variation between groups)


# In[53]:


stats.f_oneway(high,medium,low)


# In[54]:


# HenceI got p-value greater than 0.05 that means with 95% confidence, I will fail to reject the null hypothesis
# Conclusion 
#The prioritization system is not workingl incidence rate 
print("Conclusion :")
print("The prioritization system is not working ")


# # Business Problem 5

# (a). What isthe overall level of customer satisfaction?

# In[55]:


films=pd.read_csv('Films.csv')


# In[56]:


films.dtypes


# In[57]:


films.size


# In[58]:


films.head(5)


# In[59]:


films.columns


# In[60]:


films.isnull().sum()


# In[61]:


films.shape


# In[62]:


films.Gender.replace({"1": "Male","2":"Female"},inplace=  True)
films.Marital_Status.replace({'1': 'Married', '2':'Single'}, inplace=True)
films.replace('Slngle','Single', inplace=True)


# In[63]:


films


# In[64]:


films.Overall.value_counts()


# In[65]:


plt.hist(films.Overall)
plt.xlabel("Overall Satisfacction")
plt.ylabel("Frequency")
plt.title("Overall Customers Satisfaction Chart")
plt.show()


# In[66]:


# Here we can Clearly saw that Overall level of customer's satisfaction is mostly 2 and 1
# that means, on a given scale(in the question)
# Customer's Overall Satisfaction is GOOD and EXCELLENT
print("Conclusion :")
print("Customer's Overall Satisfaction is GOOD and EXCELLENT")


# b. What factors are linked to satisfaction?

# In[67]:


# Ho=NO relation between Col Overall and Sinage
# H1=Relationship is Present between col Overall and Sinage
# Taking Confidence Interval at 95%, and p-value 0.05
overall_and_sinage= pd.crosstab(index=films.Overall,columns=(films.Sinage))


# In[68]:


stats.chi2_contingency(overall_and_sinage)


# In[69]:


# HenceI got p-value smallre than 0.05 that means with 95% confidence, I will reject the null hypothesis
# Conclusion 
#Relationship is Present between col Overall and Sinage
print("Conclusion :")
print("Relationship is Present between col Overall and Sinage ")


# In[70]:


#checkin link between col Overall and Parking


# In[71]:


# Ho= NO relation between Col Overall and Parking
# H1= Relationship is Present between col Overall and Parking
# Taking Confidence Interval at 95%, and p-value 0.05


# In[72]:


overall_and_parking= pd.crosstab(index=films.Overall,columns=(films.Parking))


# In[73]:


stats.chi2_contingency(overall_and_parking)


# In[74]:


# HenceI got p-value smaller than 0.05 that means with 95% confidence, I will reject the null hypothesis
# Conclusion 
# Relationship is Present between col Overall and Parking
print("Conclusion :")
print("Relationship is Present between col Overall and Sinage ")


# c. What is the demographic profile of Film on the Rocks patrons?

# In[75]:


films


# In[76]:


films.Gender.value_counts()


# In[77]:


# taking Percentage of Genders
# Females Percentage
213/films.shape[0]*100


# In[78]:


#  Males Percentage
117/films.shape[0]*100


# In[79]:


# Taking counts of Marital_Status
films.Marital_Status.value_counts()


# In[80]:


# taking percentage of Marital_Status
# Singles Percentage
228/films.shape[0]*100


# In[81]:


# Married Percentage
100/films.shape[0]*100


# In[82]:


# Taking most likely age group
films.Age.value_counts()


# In[83]:


# Taking most likely Income group
films.Income.value_counts()


# In[84]:


print("         Demographic Profile of Film on the Rocks Patrons :")
print(" ")
print("There are Total of 213 (64.5%) Female Profiles on The Rock Patrons ")
print("There are Total of 117 (35.5%) Males Profile on The Rock Patrons")
print("There are Total of 228 (69.9%) Single Profiles on The Rock Patrons")
print("There are Total of 100 (30.1%) Married Profiles on The Rock Patrons")
print("There are More People from Age group 2 (13-30 years) on The Rock Patrons")
print("There are More People with Income group 1 (Less than $50,000) on The Rock Patrons")


# (d). In what media outlet(s) should the film series be advertised?

# In[85]:


films


# In[86]:


films.Hear_About.value_counts()


# In[87]:


# As we have mentioned the Hear About range in the Data Available Column in Business Problem as : 
    #1 = television; 2 =newspaper; 3 = radio; 4 = website; 5 = word of mouth

# And we have multiple instances where Hear_about value is given as (2,5),(3,4),(4,5)...etc
    # and many these values have 5 in it and also the maximum occurance out of these numbers is also 5
    # So, replacing (2,5),(3,4),(4,5)...etc these values with 5
    
films.Hear_About.replace({'1,5':'5','2,5':'5', '3,4':'5','3,5':'5', '4,5':'5','5,4':'5'}, inplace=True)


# In[88]:


films.Hear_About.value_counts()


# In[89]:


# Converting Hear_about col to float to make a histogram chart 
films.Hear_About=films.Hear_About.astype("float")


# In[90]:


plt.hist(films.Hear_About)
plt.xlabel('Range')
plt.ylabel('Frequency')
plt.title('Hear About Chart')
plt.show()


# In[91]:


# we have got the 5 (Word  of moth) with maximum frequency but it does not comes under media category
# So, taking the next category i.e, 4 (Website)  with second highest frequency and category 1 i.e, (Television) as the BEst media Outlet to advertise 


# In[92]:


print("The Best Media Outlets to Advertise the film series should be WEBSITE and TELEVISION")

# In[ ]:




