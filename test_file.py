# regular imports
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
from datetime import datetime, timedelta
from dateutil.parser import parse
from pandas.tseries.offsets import DateOffset

# random numbers
import random


df = pd.read_csv('data/df_for_filter.csv')
df.rename(columns={'Unnamed: 0': 'id'}, inplace=True)
print(df.sample(5))
print(df.shape, 'should be (2096800, 126)') 


def emp_length_adjuster(e):
    if e == '< 1 year':
        return 0
    if e == '10+ years':
        return 2
    else:
        return 1


def months_del_changer(x):
    if x>=200:
        return 2
    if x>60:
        return 1
    else:
        return 0

# my guess for loss is funded_amnt - total_pymnt + collection_recovery_fee 
def loss_calculate(row):
    return row['funded_amnt'] - row['total_pymnt'] + row['collection_recovery_fee'] 

def loss_percent(row):
    return row['loss_amnt'] / (row['funded_amnt'])

# defining the deal (will be modify this shortly)
def define_the_deal(df_, year):
    #looking at 36 month so we have more years to work with
    if year == 'ALL': # All is all years before 2015
        df2 = df_[(df_['issue_year']<2015) & (df_['term_60']== 0)]
    else:
        df2 = df_[(df_['issue_year']==year) & (df_['term_60']== 0)]
    good = df2[df2['loss_amnt']< 0]['id'].count()
    bad = df2[df2['loss_amnt']> 0]['id'].count()
    ugly = df2[df2['loss_amnt']== 0]['id'].count()
    total = good + bad + ugly
    status_count = df2[df2['good']==True]['id'].count()
    return good, bad, ugly, total, status_count

def the_good(df_, col_):
    
    df_new = df_.loc[df_['grade'].isin(['C', 'D', 'E', 'F'])].groupby([col_, 'good']).agg({'id':'count'}).reset_index()
    df_new['by_group'] = [df_new[df_new[col_]==x]['id'].sum() for x in df_new[col_]]
    df_new['percent'] = round(df_new['id']/df_new['by_group']*100, 2)
    print(df_new)



def improved_status(row):
    x = row['loss_percent']
    if x < -.20: # over 20% profit
        return 6
    if x < -.0999: # 10% to 19.99
        return 5
    if x < -.01: # 0 t0 9.99
        return 4
    if x < .1:
        return 3 # loss 0 to 10%
    if x < .25:
        return 2 # loss 10 to 25
    if x < .5:
        return 1 # loss 25% to 50%
    else:
        return 0 # big loss over 50%

def get_grade_count(df_, year, term, grade): 
    ''' 
        gets count of loans in dataframe based on less than year term (0 for 36 month) and grade
        returns count
    '''
    return df_[(df_['issue_year']<year) & (df_['term_60']==term) & (df_['grade']==grade)]['id'].count()


def get_percent_chance_m(df_, year, term, months, grade, month_delta=-1):
    '''
        calculates count, mean percent loss/gain (gain is -), std, and percent chance it happens
        based on less than year, term, month of loan and grade of dataframe
        inputs: dataframe, year, term, months (will do month between), grade
        returns dataframe with 
    ''' 
    ret_df = pd.DataFrame(df_[(df_['issue_year']<year) & (df_['term_60']==term) & 
        (df_['months_of_pay'].between(months + month_delta, months)) & 
        (df_['grade'] == grade
        )].groupby(['imp_status']).agg({'loss_percent': ['count', 'mean', 'std']})).reset_index()
    ret_df['percent_chance'] = [x/get_grade_count(df_, year, term, grade) for x in ret_df[('loss_percent','count')]]
    #[x/count_ for x in ret_df]
    return ret_df

df['emp_length'] = [emp_length_adjuster(x) for x in df['emp_length']]
df['mths_since_last_delinq'] = df['mths_since_last_delinq'].fillna(200)
df['mths_since_last_delinq'] = [months_del_changer(x) for x in df['mths_since_last_delinq']]

df['loss_amnt'] = df.apply(loss_calculate, axis=1) # this should be the method going forward, much cleaner
df['loss_percent'] = df.apply(loss_percent, axis=1) 

df['imp_status'] = df.apply(improved_status, axis=1)


good, bad, ugly, total, status_count = define_the_deal(df, 'ALL')
print('The Good', good, round(good/total*100, 1))
print('The Bad', bad, round(bad/total*100, 1))
print('The Ugly', ugly, round(ugly/total*100, 1))
print('Total Count', total)
print('The Status Count %', round(status_count/total*100,2))


# viewing counts to see how this looks over time
for y in [2007,2008,2009,2010, 2011, 2012,2013,2014,2015]:
    good, bad, ugly, total, status_count = define_the_deal(df, y)
    print(y)
    print('The Good', good, round(good/total*100, 1))
    print('The Bad', bad, round(bad/total*100, 1))
    print('The Ugly', ugly, round(ugly/total*100, 1))
    print('Total Count', total)
    print('The Status Count %', round(status_count/total*100,2))



df[(df['issue_year']<2015) & (df['months_of_pay']<2)].groupby(['grade', 'imp_status']).agg({'loss_amnt': ['mean', 'count']}).reset_index()

df[df['issue_year']<2015].groupby('imp_status').agg({'loss_amnt': ['mean', 'count']})



year = 2015
term = 0
grade = 'D'
months = 0


print(get_grade_count(df, year, term, grade))
print(get_percent_chance_m(df, year, term, months, grade))


print(the_good(df, 'term_60'))
print(the_good(df, 'application_type'))
print(the_good(df, 'grade'))
print(the_good(df, 'mths_since_last_delinq'))
print(the_good(df, 'fico'))
print(the_good(df, 'earliest_credit_10_more'))
print(the_good(df, 'emp_length'))
print('without joint')
print(the_good(df[(df['emp_length'].isin([2])) & (df['earliest_credit_10_more']==1) & (df['fico'].isin([4,5])) & (df['term_60']==1) & (df['home_ownership']=='MORTGAGE') & (df['application_type']!='Joint App')], 'grade'))
print('with joint')
print(the_good(df[(df['emp_length'].isin([2])) & (df['earliest_credit_10_more']==1) & (df['fico'].isin([4,5])) & (df['term_60']==1) & (df['home_ownership']=='MORTGAGE') & (df['application_type']=='Joint App')], 'grade'))

print(df[(df['months_of_pay'].isin([-1, 0]))].groupby('grade').agg({'loss_amnt': ['count', 'min', 'max', 'mean', 'sum']}))



# figure out the loss for good!

amount_fields = ['loan_amnt', 'funded_amnt'	,'funded_amnt_inv',	'int_rate',	'installment' ,'out_prncp',	'out_prncp_inv',	'total_pymnt'	,'total_pymnt_inv',	'total_rec_prncp',	'total_rec_int'	,'total_rec_late_fee',	'recoveries',	'collection_recovery_fee',	'loss_amnt', 'loss_percent']


df[(df['loan_status']=='Charged Off') & (df['grade']=='C') & (df['months_of_pay']==-1)][amount_fields].sample(10)


# my guess for loss is funded_amnt_inv - total_pymnt + collection_recovery_fee 
def loss_calculate(row):
    return row['funded_amnt_inv'] - row['total_pymnt'] + row['collection_recovery_fee'] 

df['loss_amnt'] = df.apply(loss_calculate, axis=1)
#df['loss_amnt']= [df['funded_amnt_inv'] - df['total_pymnt'] + df['collection_recovery_fee'] for _ in df['id']]


df[(df['emp_length']==2) & (df['earliest_credit_10_more']==1) & (df['fico'].isin([4,5])) & (df['mths_since_last_delinq']==2)]




# initual setup - setting up cash account
df_cash = pd.DataFrame()
df_=[]

fields_ = ['id',
           'loan_amnt',
           'funded_amnt',
           'funded_amnt_inv',
           'int_rate',
           'installment',
           'grade',
           'sub_grade',
           'issue_d',
           'out_prncp',
           'out_prncp_inv',
           'total_pymnt',
           'total_pymnt_inv',
           'total_rec_prncp',
           'total_rec_int',
           'total_rec_late_fee',
           'recoveries',
           'collection_recovery_fee',
           'last_pymnt_d',
           'last_pymnt_amnt',
           'months_of_pay',
           'good',
           'term_60'
          ]
# test to make sure the fields were in dataframe
set(fields_) - set(df.columns) 

def cash_init(start_date='20070601', start_=200):
    ''' Sets up the Cash Dataframe used to track deal progress '''
    data = {'Date': [start_date],
           'from': ['Start'],
            'cash': [start_],
            'invested': [0] 
           }
    df_cash = pd.DataFrame(data, columns = ['Date', 'from', 'cash', 'invested'])
    df_cash['Date'] = pd.to_datetime(df_cash['Date']) 
    return df_cash


def loan_picker(df, issue_year_, issue_month_, inq_last_6mths_):
    ''' based on loan paramenters, this will return a list of loan id's for month, year '''
    # make dict start
    start_date = '20071201'
    grade_ = ['C', 'D', 'E', 'F']
    fico_ = 2 # above this
    min_int_rate_ = 10
    emp_length_ = '< 1 year'
    annual_inc_ = 70000
    earliest_credit_10_more_ = 1
    loan_amnt_ = (5000, 25000)
    term_60_ = 0
    chargeoff_within_12_mths_ = 0
    purpose_ = ['credit_card', 'debt_consolidation']
    total_acc_ = 35 # less than this amount
    installment_ = 850
    init_cash = 400
    # make dict end
    df_filtered = df[(df['issue_year']==issue_year_) & # check ()...
       (df['grade'].isin(grade_)) & 
       (df['fico']>fico_) & 
       (df['int_rate']>min_int_rate_) & 
       (df['home_ownership']!='RENT') & 
       (df['emp_length']!= emp_length_) &
       (df['annual_inc'] > annual_inc_) &
       (df['earliest_credit_10_more'] == earliest_credit_10_more_) &
       (df['loan_amnt'].between(loan_amnt_[0], loan_amnt_[1])) &
       (df['term_60']==term_60_) &
       (df['purpose'].isin(purpose_)) &
       (df['total_acc'] < total_acc_) &
       (df['installment'] < installment_) &
       (df['issue_month']==issue_month_) & # .isin(issue_month_) &
       (df['inq_last_6mths']==inq_last_6mths_)   
      ].reset_index(drop=True)
    return df_filtered['id']


def loan_calc(inv, df_):
    '''inv = investement amount 
    df_ is the dataframe
    this function calculates monthly payments and amount returned based on a loan and amount invested
    '''
    I_ = []
    P_ = []
    inv_ = []
    LAST_PAYMENT = round(df_['last_pymnt_amnt'] * (inv/df_['funded_amnt_inv']), 2)
    TERM = 36 if df_['term_60']==0 else 60
    END_MNTH = min([TERM, df_['months_of_pay']+1])
    R = 1 + (df_['int_rate'])/(12*100)              # calc monthly rate
    X = inv * (R**TERM)*(1-R)/(1-R**TERM)           # calc monthly payment
    I_.append(0)
    P_.append(0)
    inv_.append(inv)
    for n_ in range(1,END_MNTH + 1):                 # goes through months, skips first month (first payment is the following month)
        I = round(inv * (R-1), 2)            # interest calculation
        if n_ == END_MNTH:                           # if last month (last month is lessor of term or months Pay field)
            X = LAST_PAYMENT                        # if last payment, get last payment from dataframe
            I = round((df_['out_prncp'] - (inv - X)), 2)
        inv = round(inv - (X-I),2)           # current invested amount is previous invested minus (Payment - interest)    
        #print((X - I, I))
        I_.append(I)
        P_.append(X - I)
        inv_.append(I - X)
    return (P_, I_, inv_, df_)

def add_cash(df_cash_, df_inv):
    return df_cash_.append(df_inv, ignore_index=True)


def invest(df_cash_, loan_,df_, P, I, inv):       
    df_investment = pd.DataFrame(columns=['Date','from', 'P', 'I', 'cash', 'invested'])
    df_investment['P'] = P
    df_investment['I'] = I
    df_investment['invested'] = inv
    df_investment['Date'] = [parse(df_['issue_d']) + DateOffset(months=i) for i in range(0, df_investment['P'].shape[0])]
    df_investment['cash'] = df_investment['P'] + df_investment['I']
    df_investment['from'] = loan_
    df_investment.loc[0,'cash'] = df_investment.loc[0, 'invested']*-1
    #print(df_investment)
    return add_cash(df_cash_, df_investment)


def run_loan(df, id_, fields_, invest_):
    df_ = df.loc[id_]#, fields_]
    #rec_back = 0
    return loan_calc(invest_, df_)

#gets cash at a date
def get_cash(date_, df_cash_):
    return round(df_cash_.loc[df_cash_['Date']==date_]['cash'].sum(),2)

def run_the_investment(df, start_date, init_cash, years, inq_last_6mths_, fields_):
    hide = 1
    df_cash_ = cash_init(start_date, init_cash)
    loans_=[]
    invested_amount = 0
    return_ = 0 
    tot_loans = 0
    win_ = 0
    R = .10/12 
    df_ = df
    min_invest = 50
    tot_loan_count = 0
    for y in years:
        for m in range (1,13):
            loans_ = (loan_picker(df, y, m, inq_last_6mths_))
            old_cash = get_cash(str(m)+'/'+str(y), df_cash_)
            Max_Loans = int((old_cash + init_cash)/min_invest)
            tot_loan_count += len(loans_)
            if not len(loans_) > 0:
                continue 
            if len(loans_) > Max_Loans:
                loans_ = pd.Series(random.sample(list(loans_), Max_Loans))
            if not hide:
                print('@'*100)
                print(m, y, len(loans_))
                print(loans_)
            for loan_ in loans_:
                invested_amount +=(init_cash/len(loans_))
                P, I, inv, df_ = run_loan(df, loan_, fields_, (init_cash + old_cash)/len(loans_))
                df_cash_ = invest(df_cash_, loan_, df_, P, I, inv)
                return_ = df_cash_[df_cash_['from']==loan_]['cash'] 
                tot_loans += 1
                if np.npv(R, return_)>0:
                    win_+=1
                if not hide:
                    print(f'NPV  :  {np.npv(R, return_) :>6.2f}')
                    print(f'IRR  :  {np.irr(return_*-1)* 12:>7.2%}')
    print('Wins :', win_, 'Total loans :', tot_loans, 'Available Loans :', tot_loan_count)
    cash_flow = df_cash_.groupby('Date')['cash'].sum()
    npv_ = np.npv(R, cash_flow)
    irr_ = np.irr((cash_flow)*-1)* 12
    print(f'NPV  :  {npv_:>6.2f}')
    print(f'IRR  :  {irr_:>7.2%}')
    return df_cash_, invested_amount, npv_, irr_, win_, tot_loans



start_date = '20071201'
init_cash = 300
years = [2012, 2013, 2014, 2015]
inq_last_6mths_ = 0


for x in range(0, 3):
    print(x+1)
    df_cash, invested_amount, NPV, IRR, WINS, TOT_LOANS = run_the_investment(
            df, 
            start_date, 
            init_cash, 
            years, 
            inq_last_6mths_, 
            fields_
            )
 df.loc[(df['issue_year']<2017) & (df['issue_year']>2010) & (df['term_60']==0)].groupby(['issue_year','grade']).agg({'int_rate': ['count', 'mean']})   



 def run_the_investment_counts(df, fields_):
    hide = 1
    df_cash_ = cash_init(start_date, init_cash)
    loans_=[]
    invested_amount = 0
    return_ = 0 
    tot_loans = 0
    win_ = 0
    R = .10/12 
    df_ = df
    min_invest = 50
    tot_loan_count = 0
    grade_ = ['D', 'E', 'F']
    fico_ = 1 # above this
    min_int_rate_ = 9.5
    emp_length_ = '< 1 year'
    annual_inc_ = 55000
    earliest_credit_10_more_ = 0
    loan_amnt_ = (10000, 25000)
    term_60_ = 0
    chargeoff_within_12_mths_ = 0
    purpose_ = ['credit_card', 'debt_consolidation']
    total_acc_ = 20 # less than this amount
    installment_ = 850
    #issue_month_ = [7]
    inq_last_6mths_ = 0
    
    years = [2013, 2014, 2015]#, 2010, 2011, 2012, 2013]
    inq_last_6mths_ = 0
    df_filtered_ = df[(df['issue_year'].isin(years)) & # check ()...
            (df['grade'].isin(grade_)) & 
            (df['fico']>fico_) & 
            (df['int_rate']>min_int_rate_) & 
            (df['home_ownership']!='RENT') & 
            (df['emp_length']!= emp_length_) &
            (df['annual_inc'] > annual_inc_) &
            (df['earliest_credit_10_more'] == earliest_credit_10_more_) &
            (df['loan_amnt'].between(loan_amnt_[0], loan_amnt_[1])) &
            (df['term_60']==term_60_) &
            (df['purpose'].isin(purpose_)) &
            (df['total_acc'] < total_acc_) &
            (df['installment'] < installment_) &
            #(df['issue_month']==issue_month_) & # .isin(issue_month_) &
            (df['inq_last_6mths']==inq_last_6mths_)   
            ].reset_index(drop=True)
    return df_filtered_.groupby(['issue_year','issue_month']).agg({'int_rate': ['count', 'mean']})
                
    
    # make dict end
run_the_investment_counts(df, fields_)
