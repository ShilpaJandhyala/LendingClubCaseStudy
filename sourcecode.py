##Problem Statement

Lending a loan is risky decision that a company takes and the borrowers who default cause the largest amount of loss to the lenders, reduces lender’s cash flow. Identifying those risky applicants helps in cutting down the amount of credit loss to the organization.
"""

#load necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

#load Dataset
loan_df=pd.read_csv("loan.csv")
loan_df.head()

#dataframe information
loan_df.info()

#checking empty columns in %age
round((100*loan_df.isnull().sum())/len(loan_df.index))

#drop all the columns which contains 100% Nan Data
loan_df.dropna(axis=1,how='all',inplace=True)
round((100*loan_df.isnull().sum())/len(loan_df.index))

#drop columns in which nan data is very high in percentage
loan_df.drop(labels=['next_pymnt_d','mths_since_last_record','mths_since_last_delinq'],axis=1,inplace=True)
loan_df.info()

loan_df.head()

#check rows which contains 100% empty data
round((100*loan_df.isnull().sum(axis=1))/len(loan_df.index))

#check the unique target values
loan_df['loan_status'].unique()

#drop Current loan status from the rows becuase it will not provide any past data it it still in running mode
loan_status_list = ['Fully Paid','Charged Off']
loan_df=loan_df[loan_df['loan_status'].isin(loan_status_list)]
loan_df.head()

#convert loan status data into numerical data i.e. Fully Paid = 0 and Charged Off = 1
loan_df['loan_status']=loan_df['loan_status'].map({'Fully Paid':0,'Charged Off':1})
loan_df.head()

#Extract necessary columns which might impact the target columns i.e. Feature selection
loan_df=loan_df.loc[:,('loan_amnt','term','int_rate','grade','emp_length',
              'home_ownership','annual_inc','issue_d','loan_status','purpose',
             'addr_state','dti')]
loan_df.head()

#create new column from the existing column
loan_df['year'] = loan_df['issue_d'].apply(lambda x:x.split('-')[1])
loan_df.head()

#convert the data type from string to int and check the type
loan_df['year']=loan_df['year'].astype(int) # Use the built-in int instead of np.int
type(loan_df['year'][0])

#draw a countplot to show the year wise increasing amount of loans in lending clubs
plt.figure(figsize=(16,7))
sb.set(style="whitegrid")
ax=sb.countplot(x='year',data=loan_df)
years = ['2007','2008','2009','2010','2011']
xpos=np.arange(len(years))
ax.set(xlabel='Years', ylabel='Total numbers of loans',title='Year-Wise Number of Loan Applicants')
plt.xticks(xpos,years)
plt.show()

"""### In this univariate analysis , Substantial growth between 2007-2011 in the number of loan applicants"""

#change the type of int_rate column from string to float
loan_df['int_rate']=loan_df['int_rate'].str.replace('%','')
loan_df['int_rate']=loan_df['int_rate'].astype(float) # Changed np.float to float
type(loan_df['int_rate'][0])

#create new column from the existing column
loan_df['month'] = loan_df['issue_d'].apply(lambda x:x.split('-')[0])
loan_df.head()

#Count plot to show the different loan purpose from the lending clubs
plt.figure(figsize=(16,7))
ax = sb.countplot(x='purpose',data=loan_df,order=loan_df['purpose'].value_counts().index)
ax.set(xlabel='Loan purpose', ylabel='No. of loan Purposes',title='Number of Loan Purpose')
plt.xticks(rotation=90)
plt.yscale('log')
plt.show()

"""### The five major purposes of Loan applications  are - ‘debt_consolidation’ , ‘credit_card’ , ‘other’ , ‘home_improvement’ & ‘major_purchase’

### debt_consolidation’ is the highest loan purpose
"""

#loan amount distribution
plt.figure(figsize=(16,7))
ax=sb.distplot(loan_df['loan_amnt'],bins=7,color='y')
ax.set(xlabel='Loan Amount',title='Distribution of loan amount')
plt.show()

"""### The distribution of loan amount i.e the amount of loan applied can be visualized from the above plot .  It has been observed that the maximum loan range amount is between 5000-10000

"""

#create a separate dataframe of defaulters
defaulters_df=loan_df.loc[loan_df['loan_status']==1]
defaulters_df.head()

#calculate the percentage of loan defaulters according to grade
grade_df=pd.DataFrame(data=round((100*defaulters_df['grade'].value_counts())/loan_df['grade'].value_counts(),2))
grade_df.rename(columns={'grade':'defaulter_perc'},inplace=True)
grade_df

"""#A bar plot showing the percentage of defaulters for each customer grade, formatted as percentages with dynamic y-axis scaling for better visualization."""

plt.figure(figsize=(16, 7))
ax = sb.barplot(x=grade_df.index, y='defaulter_perc', data=grade_df, palette="muted")
ax.set(xlabel='Customer Grade', ylabel='Percentage of Defaulters', title='Risk Analysis of Defaulters According to Grades')
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
plt.ylim(0, max(grade_df['defaulter_perc']) * 1.2)
plt.show()

"""### The percentage of Defaulters is the highest in Grade G group"""

# Calculate the percentage of loan defaulters according to loan purpose
purpose_counts = defaulters_df['purpose'].value_counts()
loan_counts = loan_df['purpose'].value_counts()
purpose_df = pd.DataFrame(
    data=round((100 * purpose_counts) / loan_counts, 2)
)
purpose_df = purpose_df.rename(columns={purpose_df.columns[0]: 'defaulter_perc'})
purpose_df.sort_values(by='defaulter_perc', ascending=False, inplace=True)
purpose_df

import matplotlib.ticker as mtick
plt.figure(figsize=(16,7))
ax=sb.barplot(x=purpose_df.index,y='defaulter_perc',data=purpose_df)
ax.set(xlabel='Customer Purpose', ylabel='Percentage of Defaulters',title='Risk Analysis of Defaulters acc. to Purpose')
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
plt.xticks(rotation=90)
plt.show()

"""### The percentage of risk of Defaulters is the largest in the Loan Purpose – ‘small_business’ , followed by ‘renewable_energy’ , ‘educational’ and so on."""

#calculate the percentage of loan defaulters according to Employee experience
emp_counts = defaulters_df['emp_length'].value_counts()
loan_counts = loan_df['emp_length'].value_counts()
emp_df = (100 * emp_counts / loan_counts).round(2)
emp_df = emp_df.to_frame(name='defaulter_perc')
emp_df.sort_values(by='defaulter_perc', ascending=False, inplace=True)
emp_df

import matplotlib.ticker as mtick
plt.figure(figsize=(16,7))
ax=sb.barplot(x=emp_df.index,y='defaulter_perc',data=emp_df)
ax.set(xlabel='Customer Exp. in Years', ylabel='Percentage of Defaulters',title='Risk Analysis of Defaulters acc. Employee Exp.')
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
plt.xticks(rotation=90)
plt.show()

"""### The percentage of risk of Defaulters is maximum for customer experience - 10+ years & minimum for 9 year’s experience . Rest year of experience is in between"""

# Calculate the percentage of loan defaulters according to home ownership
home_counts = defaulters_df['home_ownership'].value_counts()
loan_counts = loan_df['home_ownership'].value_counts()
home_df = (100 * home_counts / loan_counts).round(2)
# Convert the result to a DataFrame and name the column 'defaulter_perc'
home_df = home_df.to_frame(name='defaulter_perc')
home_df.sort_values(by='defaulter_perc', ascending=False, inplace=True)
home_df.dropna(inplace=True)
home_df

sb.set(style="darkgrid")
plt.figure(figsize=(16,6))
plt.subplot(1, 2, 1)
ax = sb.countplot(x='home_ownership',data=loan_df,order=loan_df['home_ownership'].value_counts().index)
ax.set(xlabel='HOME OWNERSHIP', ylabel='NUMBER OF PEOPLE',title='NUMBER OF PEOPLE V/S HOME OWNERSHIP')
plt.xticks(rotation=90)
plt.yscale('log')
plt.subplot(1, 2, 2)
#plt.pie(home_df['defaulter_perc'], labels = home_df.index,autopct='%.2f%%')
ax=sb.barplot(x=home_df.index,y='defaulter_perc',data=home_df)
ax.set(xlabel='HOME OWNERSHIP', ylabel='Percentage of Defaulters',title='Risk Analysis of Defaulters acc. Home Ownership')
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
plt.xticks(rotation=90)
plt.show()

"""### 1. The number of people who have home ownership -  “Rent”
### 2. The percentage of defaulter under home ownership is  “Other” followed by “Rent” and “Own” and “Mortgage”
"""

plt.figure(figsize=(16,6))
ax=sb.boxplot(x='emp_length',y='dti',data=loan_df)
ax.set(xlabel='Employee Exp.', ylabel='Debt to Income',title='DTI vs EMPLOYEE EXP')
plt.show()

"""### By doing bivariate analysis it has been observed that  10+ years of employee exp. Is having debt to income in maximum range  apart from that rest year of employee exp. are more or less in same range."""

plt.figure(figsize=(16,6))
ax=sb.boxplot(x='emp_length',y='loan_amnt',data=loan_df)
ax.set(xlabel='Employee Exp.', ylabel='Loan Amount',title='Loan Amount vs EMPLOYEE EXP')
plt.show()

"""### By doing bivariate analysis it has been observed that the employee exp. With 10+ years is taking more amount of loan and those who are less than 1 year is taken less amount of loan."""

plt.figure(figsize=(16,6))
ax=sb.boxplot(x='grade',y='int_rate',data=loan_df,)
ax.set(xlabel='Grades', ylabel='Interest Rate',title='INTEREST RATE vs EMPLOYEE GRADES')
plt.show()

"""### From the above plot it has been visualize that based on grade interest rate is going higher – “A”  is having minimum interest rate, whereas “B” is having higher interest rate."""

grade_group = loan_df.groupby('grade')
grade_group['int_rate'].median()

emp_group = loan_df.groupby('emp_length')
emp_group['loan_amnt'].median()

# Calculate the percentage of loan defaulters according to address state
state_counts = defaulters_df['addr_state'].value_counts()
loan_state_counts = loan_df['addr_state'].value_counts()
state_df = (100 * state_counts / loan_state_counts).round(2)

# Convert the result to a DataFrame and name the column 'defaulter_perc'
state_df = state_df.to_frame(name='defaulter_perc')
state_df.sort_values(by='defaulter_perc', ascending=False, inplace=True)
state_df = state_df.head(10)
state_df

import matplotlib.ticker as mtick
plt.figure(figsize=(16,7))
ax=sb.barplot(x=state_df.index,y='defaulter_perc',data=state_df)
ax.set(xlabel='Top 10 Employee State', ylabel='Percentage of Defaulters',title='Risk Analysis of Defaulters acc. Employee State (TOP 10)')
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
plt.show()

"""### The percentage of  Defaulters is maximum from NE (address) which is 60%

"""

# Calculate the percentage of loan defaulters according to loan term
term_counts = defaulters_df['term'].value_counts()
loan_term_counts = loan_df['term'].value_counts()
term_df = (100 * term_counts / loan_term_counts).round(2)

# Convert the result to a DataFrame and name the column 'defaulter_perc'
term_df = term_df.to_frame(name='defaulter_perc')
term_df.sort_values(by='defaulter_perc', ascending=False, inplace=True)
term_df

import matplotlib.ticker as mtick
plt.figure(figsize=(16,7))
ax=sb.barplot(x=term_df.index,y='defaulter_perc',data=term_df)
ax.set(xlabel='Loan Term in Months', ylabel='Percentage of Defaulters',title='Risk Analysis of Defaulters acc. Loan Term)')
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
plt.show()

"""### The percentage of Defaulters is found in 60 months as compared to 36 months – Loan Term"""
