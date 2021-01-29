import dash
import dash_core_components as dcc
import dash_html_components as html 
from dash.dependencies import Input, Output
import dash_table
import pandas as pd
import plotly.graph_objs as go


app = dash.Dash()

def to_true_false(df_, col, item=0):
    df_[col] = [0 if x == item else 1 for x in df_[col]]
    return df_


def percent_good(df_, col):
    ret = []
    
    cats_ = []
    good_ = []
    bad_ = []
    col_ = []
    tot_ = []
    int_ = []


    print('_' * 100) # printing header to seperate the column info
    print('- ' * 20 , col, ' -' * 20)
    # Column Header for Data that follows in the for loop
    print('{:30s}  {:>8s}  {:>8s}  {:>8s}  {:>14s}  {:>8s}'.format('Category', 'Good %', 'Bad %', 'Col %', 'Tot Count', 'Int %'))
    columns_ = ['Category', 'Good %', 'Bad %', 'Col %', 'Tot Count', 'Int %']


    for item_ in df_[col].unique():                               # Item is the unique category item from the column (col)
        tot = df_[col].count()                                   # total loans with data for this column
        t_ = df_[df[col]==item_][col].count()                     # Count of loans matching the category for this column
        c_ = df_[(df_[col]==item_) & (df_['good'])][col].count()  # Count of Good Loans 
        b_ = t_ - c_                                              # Count of bad loans (total - good)

        print('{:30s}  {:8.1%}  {:8.1%}  {:8.1%}  {:14,.0f}  {:8.1%}'
              .format(str(item_), c_/t_, b_/t_, t_/tot, t_, 
                      df_[df[col]==item_]['int_rate'].mean()/100))
        cats_.append(str(item_))
        good_.append(c_/t_)
        bad_.append(b_/t_)
        col_.append(t_/tot)
        tot_.append(t_)
        int_.append(df_[df[col]==item_]['int_rate'].mean()/100)
    df_ret = pd.DataFrame([good_, bad_, col_, tot_, int_], index = cats_, columns=columns_[1:])
    return df_ret



df = pd.read_csv('data/df_for_filter.csv')
df = to_true_false(df, 'tax_liens')
df = to_true_false(df, 'disbursement_method', 'cash')
df = to_true_false(df, 'emp_length', '10+ years')
df.rename(columns={'disbursement_method': 'disb_direct', 'emp_length': 'emp_len_under_10'}, inplace=True)
df_grouped = df.groupby('grade')['loan_amnt'].count()
colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}
app.layout = html.Div(style={'backgroundColor': colors['background']}, children=[
    html.H1(
        children='Lending Club - Your Cash Generating Machine',
        style={
            'textAlign': 'center',
            'color': colors['text']
        }
    ),
    html.H2(children='Question, can you use Lending Club to create monthly cash with little risk?', style={
        'textAlign': 'center',
        'color': colors['text']
    }),
    dcc.Markdown(''' ---Note---
    
        What goes into FICO scores? A popular FICO score chart describes the main 
        factors that affect score are 35% payment history, 30% debt owed, 15% age of credit history, 
        10% new credit, and 10% types of credit.''', style={
        'textAlign': 'center',
        'color': colors['text']
    }),
    html.Label('Loan Grade Checkbox', style={
        'color': 'white'
    }),
    dcc.Checklist(
        options=[
            {'label': 'Grade A', 'value': 'A'},
            {'label': 'Grade B', 'value': 'B'},
            {'label': 'Grade C', 'value': 'C'},
            {'label': 'Grade D', 'value': 'D'},
            {'label': 'Grade E', 'value': 'E'},
            {'label': 'Grade F', 'value': 'F'},
            {'label': 'Grade G', 'value': 'G'}
        ],
        value=['D', 'C'],
        style={'color': 'white'}
    ),
    html.Label('Year Checkbox', style={
        'color': 'white'
    }),
    dcc.Checklist(
        options=[
            {'label': ' 2007 ', 'value': '2007'},
            {'label': ' 2008 ', 'value': '2008'},
            {'label': ' 2009 ', 'value': '2009'},
            {'label': ' 2010 ', 'value': '2010'},
            {'label': ' 2011 ', 'value': '2011'},
            {'label': ' 2012 ', 'value': '2012'}
        ],
        value=['2009', '2010'],
        style={'color': 'white'}),
    #dcc.Input(id='my-id', value='Dashy-Man', type='text'),
    #html.Div(id='my-div', style={'color': 'white'}),
    dcc.Dropdown(
        id='dd_column',
        options=[{'label': i, 'value': i} for i in df.columns],
        value='grade'
    ),
    html.Div(id='by_column', style={'color': 'white'}),
    
    dcc.Graph(
        id='Graph1',
        figure={
            'data': [
                #{'x': df_grouped, 'y': df_grouped['loan_amnt'], 'type': 'bar', 'name': 'Good'},
                {'x': [1, 2, 3], 'y': [2, 4, 5], 'type': 'bar', 'name': u'Montréal'},
            ],
            'layout': {
                'plot_bgcolor': colors['background'],
                'paper_bgcolor': colors['background'],
                'font': {
                    'color': colors['text']
                }
            }
        }
    )
])

#@app.callback(
#    Output(component_id='my-div', component_property='children'),
#    [Input(component_id='my-id', component_property='value')]
#)

@app.callback(
    Output(component_id='by_column', component_property='children'),
    [Input(component_id='dd_column', component_property='value')]
)

def update_output_div(input_value):
    df_ = percent_good(df, input_value)
    # fig = go.Figure(data=[go.Table(header=dict(values=df_[0]),
    #    fill_color='paleturquoise', align='left',
    #   cells=dict(values=[df_.Category, df_['good %'], df_['Bad %']], fill_color='lavernder', align='left'))])
    return df_

if __name__ == '__main__':
    app.run_server(debug=True)
    