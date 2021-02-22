import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
from random import *
from functools import reduce


TITLE_TEXT_STYLE = {
    'textAlign': 'center',
    'color': '#0074D9',
    'font-size':'200%'
}

def amnt_val(perc_, amnt_g, amnt_b, adj=0):
    x = random()
    if x <= perc_:
        return amnt_g
    else:
        return amnt_b


def run_it(df_s, grade, adj=0):
    if len(grade)==1:
        grade_type = 'grade'
    else:
        grade_type = 'sub_grade'
    grades_total_l=[]
    grades_good_l=[]
    grades_g_perc_l=[]
    grades_perc_l=[]
    grades_num_loans = (df_s[(df_s[grade_type]==grade)]['grade'].count())
    max_ = df_s[(df_s[grade_type]==grade)]['months_of_pay'].max()
    #emp_adj = create_per(grade, emp)
    for m_ in range(0, 39):
        gt_count = (df_s[(df_s['months_of_pay']==m_)
                                    & (df_s[grade_type]==grade)]
                              ['grade'].count())
        gg_count = (df_s[(df_s['months_of_pay']==m_)
                                    & (df_s[grade_type]==grade)
                                   & (df_s['good']==True)]
                              ['grade'].count())
        try:
            amnt_g = df_s[(df_s['months_of_pay']==m_)
                                    & (df_s[grade_type]==grade)
                                   & (df_s['good']==True)]['balance_b100'].sample()
        except:
            try:
                amnt_g = df_s[(df_s['months_of_pay']==m_-1)
                                    & (df_s[grade_type]==grade)
                                   & (df_s['good']==True)]['balance_b100'].sample()
            except:
                amnt_g = df_s[(df_s['grade']==grade[0])
                                   & (df_s['good']==True)]['balance_b100'].sample()
        try:
            amnt_b = df_s[(df_s['months_of_pay']==m_)
                                    & (df_s[grade_type]==grade)
                                   & (df_s['good']==False)]['balance_b100'].sample()
        except:
            try:
                amnt_b = df_s[(df_s['months_of_pay']==m_-1)
                                    & (df_s[grade_type]==grade)
                                   & (df_s['good']==False)]['balance_b100'].sample()
            except:
                amnt_b = df_s[(df_s['grade']==grade[0])
                                   & (df_s['good']==False)]['balance_b100'].sample()
        grades_total = gt_count
        grades_good = gg_count
        grades_g_perc = gg_count/gt_count
        grades_perc = gt_count/grades_num_loans
        if m_ == 36:
            final_p = grades_g_perc
        x = random()
        if (x >= grades_perc):
            if m_ > 37:
                return 36, (amnt_val(final_p, amnt_g.item(), amnt_b.item())*-1)
            continue
        else:
            return m_, (amnt_val(grades_g_perc, amnt_g.item(), amnt_b.item())*-1)


def multi_run_it(df_s, num_runs, adj=0):
    x=[]
    y=[]
    z=[]
    for grd in df_s['grade'].unique():
        for _ in range(0, num_runs):
            l= len(grd)
            mon_, amnt_ret = run_it(df_s, grd)
            x.append(mon_)
            y.append(amnt_ret)
            z.append(grd)
    number_below_zero = reduce(lambda sum, j: sum  + (1 if j <= 0 else 0), y, 0)
    #sns.scatterplot(x=x, y=y, hue=z)
    print('Return %',round(sum(y)/len(y), 3),
          'Annual %',round(sum(y)/(len(y))/(sum(x)/len(x))*12,2),
          'Return Amnt', round(sum(y),2),
          'Max Loss', round(min(y),2),
          'Max Gain', round(max(y),2),
          'N Mths',round((sum(x)/len(x)),2),
          'Good', len(y)-number_below_zero,
          'Bad', number_below_zero)
    return x,y,z





def sim_child(df_,num=5):
    df_s = df_[['grade', 'sub_grade',
            'months_of_pay', 'funded_amnt', 'int_rate',
            'installment_b100', 'total_pymnt_b100', 'balance_b100',
            'loan_status', 'good',
            'emp_length', 'home_ownership',
            'verification_status', 'annual_inc_bin',
            'purpose', 'dti', 'joint', 'fico', 'issue_year', 'term_60']]
    GRADES = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    SUB_GRADES = [x for x in df_s['sub_grade'].unique()]
    df_s.sort_values('months_of_pay').reset_index(inplace=True)
    print('go')
    x, y, z = multi_run_it(df_s, num)
    new_y = [(abs(y_)/5)+1 for y_ in y]
    fig = px.scatter(x=x, y=y,
           color=z,
           width=1200, height=800,
           size=new_y,
           size_max=25,
           template="seaborn",
          title = 'Lending Club Loan Month of Return and Total Return',
          labels={'x':'Month the Loan Ended',
                 'y':'Profit/Loss Amount',

                 'color': 'Loan Grade'})
    fig.update_layout({'plot_bgcolor': 'lightgrey',
                         'paper_bgcolor':'grey',
                         'title_x': 0.5,
                         'font': {'size': 20},
                         'title': 'Survival Plot'})
    #fig.show()
    return fig
