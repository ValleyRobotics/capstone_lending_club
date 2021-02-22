import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table as dt
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objs as go
import plotly.figure_factory as ff
from myFunc import *
from main_child import main_child
from sim_child import multi_run_it



import numpy as np

# bool for loading from hard drive or dropbox
dropBox_yes=False
#load from dropbox
if dropBox_yes:
    print('True using dropbox to get data')
    print('big file downloading')
    df = pd.read_pickle('https://www.dropbox.com/s/h79jd1kkdlawkd7/lending_club_binned_data_1.pkl?dl=1')
    print('Four smaller files')
    df_accepted4a_36m_no_0_cox = pd.read_csv('https://www.dropbox.com/s/hw8muj8m0o6ny20/df_accepted4a_36m_no_0_cox.csv?dl=1')
    df_accepted4a_60m_no_0_cox= pd.read_csv('https://www.dropbox.com/s/3ijl1w85kcs2iwy/df_accepted4a_60m_no_0_cox.csv?dl=1')
    print('done with download')
else: #load from hard drive
    print('false, loading from hard drive hopefully files are downloaded')
    # with time, will add a try function here to look if not download files to data folder in app then try...
    df = pd.read_pickle('data/df_reduced_LC.pkl')
    df_accepted4a_60m_no_0_cox = pd.read_pickle('data/df_accep_60m.pkl')
    df_accepted4a_36m_no_0_cox = pd.read_pickle('data/df_accep_36m.pkl')

x_features = ['grade',
            'sub_grade',
            'emp_length',
            'fico',
            'months_of_pay',
            'issue_d',
            'issue_month',
            'balance_b100', 'balance_b100-',
            'term_60', 'good', 'years_pay']

y_features = ['grade',
            'sub_grade',
            'emp_length',
            'fico',
            'months_of_pay',
            'issue_d',
            'issue_month',
            'balance_b100', 'balance_b100-',
            'term_60', 'good', 'years_pay',
            'loan_status_bool2',
            'loan_status_bool3']

z_features = ['grade',
            'sub_grade',
            'emp_length',
            'fico',
            'months_of_pay',
            'issue_d',
            'issue_month',
            'balance_b100', 'balance_b100-',
            'term_60', 'good',
            'loan_status',
            'years_pay',
            'loan_status_bool2',
            'loan_status_bool3']

x_fields = ['LOAN_AMNT_BINNED1',
            'GRADE',
            'EMP_LENGTH_BINNED1',
            'FICO_BINNED1',
            'HOME_OWNERSHIP_OWN_IS_0']

# the style arguments for the sidebar.
SIDEBAR_STYLE = {
    'position': 'fixed',
    'top': 0,
    'left': 0,
    'bottom': 0,
    'width': '20%',
    'padding': '20px 10px',
    'font-size':'140%'#,
    #'background-color': '#f8f9fa'
}

# the style arguments for the main content page.
CONTENT_STYLE = {
    'margin-left': '25%',
    'margin-right': '5%',
    'top': 0,
    'padding': '20px 10px'
}

TEXT_STYLE = {
    'textAlign': 'center',
    'color': '#191970',
    'font-size':'140%'
}

CARD_TEXT_STYLE = {
    'textAlign': 'center',
    'color': '#0074D9'
}

CARD_TEXT_STYLE2 = {
    'textAlign': 'left',
    'color': 'white',
    'font-size':'145%'
}
TITLE_TEXT_STYLE = {
    'textAlign': 'center',
    'color': '#0074D9',
    'font-size':'200%'
}
TAB_STYLE = {
    'textAlign': 'center',
    'color': 'green',
    'background-color':'yellow',
    'font-size':'150%'
}
#End css
controls = return_control(x_features, y_features, z_features)

ext_stylesheets=[dbc.themes.SLATE]

app = dash.Dash(external_stylesheets=ext_stylesheets)

sidebar = html.Div(
    [
        html.H1('Parameters', style=TITLE_TEXT_STYLE),
        html.Hr(),
        controls
    ],
    style=SIDEBAR_STYLE,
)
content = html.Div([dcc.Tabs([
dcc.Tab(label='Main', children=main_child()),
# EDA Tab
dcc.Tab(label='EDA', children=[
html.H1('Lending Club EDA Dashboard', style=TITLE_TEXT_STYLE),
html.H3('Loan years beyond 2015 will not be accurate because they are still in valid term length. \
            Prior to 2009, Lending Club used different evaluation method for rating loans. \
            Loan balances have been scaled to $100 based on loan_funded_amnt.'),
html.Div([
],
style={'width':'48%', 'display':'inline-block'}),
dcc.Graph(id='feature-graphic', style={'width':'100%', 'height':800}),


html.Div(dbc.Card([
    dcc.Graph(id='textArea', style={'display': 'inline-block','width':'50%', 'height':700}),
    #dcc.Textarea(id='textArea',
                # value='testing',
                # style={'fontSize':20, 'width': '50%', 'height':700,
                # 'display':'inline-block',
                # 'backgroundColor':'grey',
                # 'textAlign': 'center'}),
    dcc.Graph(id='second-graph',style={'width':'50%', 'height':700,
    'display':'inline-block',
    'textAlign': 'center'})
],style={'width':'100%', 'display':'inline-block'}))

], style={"backgroundColor": "lightgrey"}),
# Machine Learning Tab ################################################################################
dcc.Tab(label='EDA - Survival', children=[
    html.Div([
        html.Div([dcc.Tabs([
        ###
        dcc.Tab(label='Survival Curves1', children=[
        html.Div([
            html.H1('Competing Risks: Grade (36 month vs 60 month)', style=TITLE_TEXT_STYLE),
            html.Img(src=app.get_asset_url('my-grade_36m.png'),width="100%", height="100%"),# '/assets/grade_36m.png'))"),
            html.Img(src=app.get_asset_url('my-Screen Shot-Records-COMPETING_RISKS2_grade_60m.png')
                                            ,width="100%", height="100%"),

        ],style={"backgroundColor": "lightgrey"}
        )]),
        dcc.Tab(label='Survival Curves2', children=[
        html.H3('Competing Risks: FICO Score (36 month vs 60 month)', style=TITLE_TEXT_STYLE),
            html.Img(src=app.get_asset_url('my-Screen Shot-Records-COMPETING_RISKS2_fico_36m.png'),width="100%", height="100%"),
            html.Img(src=app.get_asset_url('my-Screen Shot-Records-COMPETING_RISKS2_fico_60m.png'),width="100%", height="100%"),
        ], style={"backgroundColor": "lightgrey"}),
        dcc.Tab(label='Survival Curves3', children=[
            html.H3('Competing Risks: FICO Score (36 month vs 60 month)', style=TITLE_TEXT_STYLE),
            html.Img(src=app.get_asset_url('my-Screen Shot-Records-COMPETING_RISKS2_int_rate_36m.png'),width="100%", height="100%"),
            html.Img(src=app.get_asset_url('my-Screen Shot-Records-COMPETING_RISKS2_int_rate_60m.png'),width="100%", height="100%"),
             ], style={"backgroundColor": "lightgrey"}),
             dcc.Tab(label='Survival Curves4', children=[
             html.H3('Competing Risks: Employment Length (36 month vs 60 month)', style=TITLE_TEXT_STYLE),
             html.Img(src=app.get_asset_url('my-ScreenShotCOMPETING_RISKS2_emp_length_36m.png'),width="100%", height="100%"),
             html.Img(src=app.get_asset_url('my-Screen Shot-Records-COMPETING_RISKS2_emp_length_60m.png'),width="100%", height="100%"),
             ], style={"backgroundColor": "lightgrey"}),


            dcc.Tab(label='Survival Models', children=[
                 html.Div([
                     html.H1('Survival a Way of Life ', style=TITLE_TEXT_STYLE)
                 ], style={"backgroundColor": "lightgrey"}),
                 html.P("Features"),
                 dbc.RadioItems(id='boxplot_fields',
                                options=[{'label': i, 'value': i} for i in x_fields],

                                value='LOAN_AMNT_BINNED1',
                                inline=True
                 ),
                 dcc.Graph(id="out_boxplot_fields_36", style={'width':'100%', 'height':600}),

                 dcc.Graph(id="out_boxplot_fields_60", style={'width':'100%', 'height':600}),
                 ], style={"backgroundColor": "lightgrey"},
                 ),
            ####
            dcc.Tab(label='Time-Dependent EDA: P&L Trends (36mnth vs 60mnth) / Survival Analysis', children=[
                 html.Div([
                     html.H1('P&L: ', style=TITLE_TEXT_STYLE)
                 ], style={"backgroundColor": "lightgrey"}),
                 html.P("Loan Status:"),
                 dcc.RadioItems(id='boxplot_loan_status',
                                options=[{'label': i, 'value': i} for i in ['Default/Delinquent', 'Prepaid']],
                                labelStyle={'display': 'inline-block'},
                                value='Prepaid'
                 ),

                 dcc.Graph(id="out_boxplot_loan_status", style={'width':'100%', 'height':800}),
             ])], style={"backgroundColor": "lightgrey"})
             ])])]),

#######################################################################################################
dcc.Tab(label='Sim', children=[
    html.H1('Lending Club Loan Simulator', style=TITLE_TEXT_STYLE),
    html.H3('Loan Simulator will run different deals to calculate a potential profit or loss.\
            When multiple deals are done, an estimate can be done on the expected return range for the deal.'),
    html.Div([html.Br(),
    html.Div(
        className="row", children=[
    dbc.Card([
    html.Label('Num of Loans per Grade'),
    dbc.RadioItems(
        id='num_of_runs',
        options=[{'label': i, 'value': i} for i in [1,5,10,20,50,100,250,500,1000]],
        value=5,
        inline=False,
        style={
            'margin': 'auto'
        }),
    ], style={'display': 'inline-block', 'width': '15%'}),
    ####
    dbc.Card([
        html.Label('Employment Length Years'),
        dbc.Checklist(
        id='empl_items',
        options=[
        {
            'label': 'Length = 0',
            'value': 0
        },
        {
            'label': 'Length < 1',
            'value': 1
        },
        {
            'label': 'Length 1-4',
            'value': 2
        },
        {
            'label': 'Length 5 - 8',
            'value': 3
        },
        {
            'label': 'Empl Length 9+',
            'value': 4
        }
        ],
        value=[0,1,2,3,4],
        inline=False,
        style={
            'margin': 'auto'
        }
    )], style={'display': 'inline-block', 'width': '20%'}

    ),
    dbc.Card([
        html.Label('Fico Score'),
        dbc.Checklist(
            id='fico_items',
            options=[{
                'label': 'Score 0',
                'value': 0
                },
                {
                    'label': 'Score 1',
                    'value': 1
                },
                {
                    'label': 'Score 2',
                    'value': 2
                },
                {
                    'label': 'Score 3',
                    'value': 3
                },
                {
                    'label': 'Score 4',
                    'value': 4
                },
                {
                    'label': 'Score 5',
                    'value': 5
                }
            ],
            value=[0,1,2,3,4,5],
            inline=False,
            style={
                'margin': 'auto'
            }
        )], style={'display': 'inline-block', 'width': '15%', 'padding':'5px'}

        ),
        dbc.Card([
        html.Label('Home Ownership'),
        dbc.Checklist(
            id='home_items',
            options=[{
                'label': 'Own',
                'value': 'OWN'
                },
                {
                    'label': 'Mortgage',
                    'value': 'MORTGAGE'
                },
                {
                    'label': 'Rent',
                    'value': 'RENT'
                },
                {
                    'label': 'Other',
                    'value': 'OTHER'
                },
                {
                    'label': 'None',
                    'value': 'NONE'
                }
            ],
            value=['RENT', 'MORTGAGE', 'OWN', 'OTHER', 'NONE'],
            inline=False,
            style={
                'margin': 'auto'
            }
        )], style={'display': 'inline-block', 'width': '15%', 'padding':'5px'}
        ),
        dbc.Card([
            html.Label('Funded Amount'),
            dcc.RangeSlider(
                id='fund_items',
                min=0,
                max=45000,
                step=5000,
                value=[0, 45000],
                marks={i:i for i in range(0,45001, 10000)}
            ),
            html.Label('Income Amount'),
            dcc.RangeSlider(
                id='inc_slider',
                min=0,
                max=200000,
                step=25000,
                value=[0, 200000],
                marks={i:i for i in range(0,200001, 50000)}
            ),
            html.Label('File Name   '),
            dcc.Input(id="file", type="text", placeholder="sim_file.csv", debounce=True),

            dbc.RadioItems(
                id='file_items',
                options=[
                {
                    'label': 'Run As Simulator',
                    'value': 'RUN'
                },
                {
                    'label': 'Load File',
                    'value': 'LOAD'
                }
                ],
                value='RUN',
                inline=True,
                style={
                    'margin': 'auto'
                }),
                ], style={'display': 'inline-block', 'width': '35%', 'padding':'5px'}
                )
        ],style=dict(display='flex')),

        html.Br(),
        dbc.Button(
            id='simit_button',
            n_clicks=0,
            children='Simit',
            color='primary',
            block=True
        ),
        ], style={'width':'98%', 'display':'inline-block', 'padding':'25px'}),

    html.Div(dbc.Card([
        dcc.Graph(id='scatter-graphic', style={'width':'100%', 'height':800})
    ]), style={"backgroundColor": "lightgrey"}),
    html.Div(dbc.Card([
        dcc.Graph(id='box-graph', style={'width':'100%', 'height':600})
        ]), style={"backgroundColor": "lightgrey"}),
    html.Div(dbc.Card([
    html.H3('Lending Club Simulated Data Table'),
    html.Div([
    dcc.Graph(id='dataframe', style={'display': 'inline-block','width':'80%', 'height':400}),
    dcc.Graph(id='dataframe_mean', style={'display': 'inline-block','width':'20%', 'height':400})]),
    ]), style={"backgroundColor": "lightgrey"}),
     ]),
dcc.Tab(label='Conclusion', children=[
html.Div([
    html.H1('The Big Lending Club Conclusion', style=TITLE_TEXT_STYLE),
    html.H3('They should go out of business!'),
    html.P('I only say that because they are going out of business!')
    ])
], style={"backgroundColor": "lightgrey"})
],style=TAB_STYLE)],style=CONTENT_STYLE)

app.layout = html.Div([sidebar, content])

# running the sim
@app.callback([Output('scatter-graphic', 'figure'),
                Output('box-graph', 'figure'),
                Output('dataframe', 'figure'),
                Output('dataframe_mean', 'figure')],
                [Input('simit_button', 'n_clicks')],
                 [State('year_slider', 'value'),
                 State('grade_items', 'value'),
                 State('term_items', 'value'),
                 State('num_of_runs', 'value'),
                 State('fico_items', 'value'),
                 State('empl_items', 'value'),
                 State('home_items', 'value'),
                 State('fund_items', 'value'),
                 State('inc_slider', 'value'),
                 State('file', 'value'),
                 State('file_items', 'value')
                  ])
def update_graph(n_clicks,
                    year_slider_value,
                    grade_items_value,
                    term_items_value,
                    num_of_runs_value,
                    fico_items_value,
                    empl_items_value,
                    home_items_value,
                    fund_items_value,
                    inc_slider_value,
                    file_value,
                    file_items_value,
                    ):
    if file_items_value=='RUN':
        df_sim = df[(df['issue_year'].between(year_slider_value[0], year_slider_value[1]))]
        df_sim = df_sim[(~df_sim['loan_status'].isin(['Current', 'Not Current']))]
        df_sim = df_sim[df_sim['grade'].isin(grade_items_value)]
        df_sim = df_sim[df_sim['term_60'].isin(term_items_value)]
        df_sim = df_sim[df_sim['emp_length'].isin(empl_items_value)]
        df_sim = df_sim[df_sim['fico'].isin(fico_items_value)]
        df_sim = df_sim[df_sim['home_ownership'].isin(home_items_value)]
        df_sim = df_sim[df_sim['funded_amnt'].between(fund_items_value[0],fund_items_value[1])]
        if inc_slider_value[1]>=200000:
            max = 200000000
        else:
            max = inc_slider_value[1]
        df_sim = df_sim[df_sim['annual_inc'].between(inc_slider_value[0],max)]
        df_s = df_sim[['grade', 'sub_grade',
                'months_of_pay', 'funded_amnt', 'int_rate',
                'installment_b100', 'total_pymnt_b100', 'balance_b100', 'annual_inc',
                'loan_status', 'good',
                'emp_length', 'home_ownership',
                'verification_status', 'annual_inc_bin',
                'purpose', 'dti', 'joint', 'fico', 'issue_year', 'term_60']]
        df_s.sort_values('months_of_pay').reset_index(inplace=True)
        print('go')
        x, y, z = multi_run_it(df_s, num_of_runs_value)
    else:
        print('Working on this')
    new_y = [(abs(y_)/5)+1 for y_ in y]
    print('graph coming')
    fig = px.scatter(x=x, y=y,
           color=z,
           size=new_y,
           size_max=25,
           #template="seaborn",
          title = 'Lending Club Simulated - Number of Months vs. Profit Loss of Loan',
          labels={'x':'Month the Loan Ended',
                 'y':'Profit/Loss Amount',
                 'color': 'Loan Grade'})
    fig.update_layout({'plot_bgcolor': 'lightgrey',
                     'paper_bgcolor':'grey',
                     'title_x': 0.5,
                     'font': {'size': 20},
                     'font_color': 'yellow'})
    data_tuples = list(zip(x,y,z))
    sim_results = pd.DataFrame(data_tuples, columns=['Months', 'Profit', 'Grade'])
    sim_results.sort_values('Grade', inplace=True)
    sim_data = sim_results.groupby('Grade').agg({'Profit':['count', 'sum',
                                                'mean', 'min', 'max', np.median, 'std'],
                                                'Months':['mean',np.median, 'std']}).round(2).reset_index()
    sim_data_mean = sim_data.mean(numeric_only=True).round(2).reset_index()

    sim_data_mean['Name'] = sim_data_mean['level_0'] + '_' + sim_data_mean['level_1']
    sim_data_mean.drop(['level_0', 'level_1'], axis=1,inplace=True)
    sim_data_mean.columns = ['Value', 'Name']

    print(sim_data_mean)
    sim_data.columns = ['Grade', 'Count', 'Profit Sum', 'Profit Mean', 'Profit Min', 'Profit Max', 'Profit Median', 'Profit STD', 'Mnth Mean', 'Mnth Median', 'Mnth STD']
    fig_table= ff.create_table(sim_data)
    fig_table_mean = ff.create_table(sim_data_mean)
    file_save(sim_data, file_value)
    fig1 = px.box(sim_results, y='Profit', x='Grade', color='Grade',
                    title='Lending Club Simulated Box Plots',
                    labels={'x':'Month the Loan Ended',
                           'y':'Profit/Loss Amount',
                           'color': 'Loan Grade'})
    fig1.update_layout({'plot_bgcolor': 'lightgrey',
                     'paper_bgcolor':'grey',
                     'title_x': 0.5,
                     'font': {'size': 20},
                     'font_color': 'yellow'})

    ####
    return fig, fig1, fig_table, fig_table_mean
    ## end the sim



@app.callback([Output('feature-graphic', 'figure'),
                Output('second-graph', 'figure'),
                Output('textArea', 'figure'),
                Output('out_boxplot_loan_status', 'figure'),
                Output('out_boxplot_fields_36', 'figure'),
                Output('out_boxplot_fields_60', 'figure')],
                [Input('submit_button', 'n_clicks'),
                Input('boxplot_fields', 'value')],
                 [State('year_slider', 'value'),
                 State('radio_t', 'value'),
                 State('radio_l', 'value'),
                 State('radio_r', 'value'),
                 State('grade_items', 'value'),
                 State('term_items', 'value'),
                 State('xaxis', 'value'),
                 State('yaxis', 'value'),
                 State('zaxis', 'value')

                  ])
def update_graph(n_clicks,
                    boxplot_fields_value,
                    year_slider_value,
                    radio_t,
                    radio_l_value,
                    radio_r_value,
                    grade_items_value,
                    term_items_value,
                    xaxis_name,
                    yaxis_name,
                    zaxis_name
                    ):
    df_ = df[(df['issue_year'].between(year_slider_value[0], year_slider_value[1]))]
    df_ = df_[(~df_['loan_status'].isin(['Current', 'Not Current']))]

    df_ = df_[df_['grade'].isin(grade_items_value)]
    df_ = df_[df_['term_60'].isin(term_items_value)]

    df_.sort_values('sub_grade', inplace=True)



    if radio_t == 'BC_rd':
        df_sub_g = df_[df_['good']==True].groupby(xaxis_name).agg({'loan_amnt':'count'})
        df_sub_b = df_[df_['good']==False].groupby(xaxis_name).agg({'loan_amnt':'count'})

        text_ = df_.groupby(xaxis_name)[xaxis_name].count()

        df_sub_g = df_sub_g[df_sub_g['loan_amnt'] > 0]
        df_sub_b = df_sub_b[df_sub_b['loan_amnt'] > 0]


        traceGood = go.Bar(x=df_sub_g.index, y=df_sub_g.loan_amnt, name='Good',
                            hovertemplate = '<b>GOOD</b>: %{y:,-2f}')
        traceBad = go.Bar(x=df_sub_b.index, y=df_sub_b.loan_amnt, name='Bad',
                            hovertemplate = '<b>BAD</b>: %{y:,-2f}')

        fig1 = {'data':[traceGood, traceBad],
                'layout': go.Layout(
                title='Good Loans Vs Bad Loans {} '.format(xaxis_name),
                xaxis_title=xaxis_name,
                yaxis_title="Count of Loans",
                paper_bgcolor = 'grey',
                font=dict(size=18),

                hoverlabel=dict(
                    bgcolor="white",
                    align = 'right',
                    font_size=16,
                    font_family="Rockwell"),
                plot_bgcolor='rgb(10,10,10)', barmode='stack',
                font_color='yellow')}


    if radio_t == 'DC_rd':
        df_sx = df_.groupby([xaxis_name, zaxis_name])['loan_amnt'].count().reset_index()
        fig1 = px.histogram(df_sx[df_sx['loan_amnt']>0],
                            x=xaxis_name,

                            y='loan_amnt',
                            color=zaxis_name,
                            #marginal_y='box',
                            marginal='box',
                            title='Lending Club Loan Status Density')
    if radio_t == 'SC_rd':

        fig1 = px.scatter(df_[df_['loan_amnt']>0],
                            x=xaxis_name,
                            y=df_['balance_b100'] * -1,
                            color=zaxis_name,
                            title='Lending Club Loan Status Scatter',
                            labels={'x':xaxis_name,
                                   'y':'Profit/Loss Amount',
                                   'color': zaxis_name})


    if radio_t == 'KDE_rd':
        #
        fig1 = ff.create_distplot(
                [df_[xaxis_name][df_['grade'] == c].values
                 for c in df_.grade.unique()
                ],
                df_.grade.unique(),
                show_hist=False,
                show_rug=False
            )
        for d in fig1['data']:
            d.update({'fill': 'tozeroy'})
    if radio_t=='SSC_rd':
        fig1 = return_survival(df_, grade_items_value)

    if radio_t != 'BC_rd':
        fig1.update_layout({'plot_bgcolor': 'lightgrey',
                            'paper_bgcolor':'black',
                            'title_x': 0.5,
                            'font': {'size': 20},
                            'font_color': 'yellow'})
    if radio_l_value=='DF_rd':
        line_text, ret_data = (percent_good(df_.sort_values('sub_grade'), xaxis_name))
        ###
        df_data = pd.DataFrame(ret_data, columns=['Category', 'Good%', 'Bad%', 'Col%', 'Total', 'Int%'])
        print(df_data.head())
        line_text= ff.create_table(df_data)
        
        ###
    elif radio_l_value=='NOTE_rd':
        line_text = 'Notice the yield curve change as you decrease the top of the years range to less than 2016.'\
                    'This is due to the number of Current and Not Current loans still being active and their totals'\
                    'are impacting the calculation of the returns. This is more prevelientt when including term 60 loans'
    else:
        line_text = 'Under Construction, please drive safe and enjoy your stay!'

    if radio_r_value=='sb_rd':
        if xaxis_name == 'grade':
            df_sx = df_.groupby(['grade', 'good'])['loan_amnt'].count().reset_index()
            fig = px.sunburst(df_sx,
                                path=['grade','good'],
                                values='loan_amnt',
                                title='Lending Club Loan Status')
        else:
            df_sx = df_.groupby(['grade', xaxis_name, 'good'])['loan_amnt'].count().reset_index()
            fig = px.sunburst(df_sx,
                                path=['grade',xaxis_name,'good'],
                                values='loan_amnt',
                                title='Lending Club Loan Status')


    if radio_r_value=='yc_rd':
        loan_amnt =df_.groupby(xaxis_name)['balance_b100'].count()*100
        rec_amnt = df_[df_['issue_year']<=2014].groupby(xaxis_name)['balance_b100'].mean()
        yield_curve = ((rec_amnt*-1))
        fig = px.line(yield_curve,
                        x=yield_curve.index,
                        y=yield_curve,
                        title='Average Return on $100')
    if radio_r_value=='sc_rd':
        fig = return_survival(df_, grade_items_value)

    fig.update_layout({'plot_bgcolor': 'lightgrey',
                        'paper_bgcolor':'grey',
                        'title_x': 0.5,
                        'font': {'size': 20},
                        'font_color': 'yellow'})
    ####
    x_fields = ['LOAN_AMNT_BINNED1',
    'GRADE',
    'EMP_LENGTH_BINNED1',
    'FICO_BINNED1',
    'HOME_OWNERSHIP_OWN_IS_0']

    y = 'profit_&_loss'

    df_accepted4a_36m_no_0_cox.sort_values(boxplot_fields_value, inplace=True)
    fig36_SL = px.box(df_accepted4a_36m_no_0_cox, y=y, x=boxplot_fields_value, color=boxplot_fields_value,
                labels={boxplot_fields_value: boxplot_fields_value,
                       y: y,
                       'loan_status': 'Loan Status'},
                    title=str(boxplot_fields_value) + ' vs. Profit & Loss: 36 Months',
                 template= 'seaborn')
    fig36_SL.update_layout({'plot_bgcolor': 'lightgrey',
                 'paper_bgcolor':'grey'})

    df_accepted4a_60m_no_0_cox.sort_values(boxplot_fields_value, inplace=True)
    fig60_SL = px.box(df_accepted4a_60m_no_0_cox, y=y, x=boxplot_fields_value, color=boxplot_fields_value,
                #notched=True,
                labels={boxplot_fields_value:boxplot_fields_value,
                       y:y,
                       'loan_status': 'Loan Status'},
                 title= str(boxplot_fields_value) + ' vs. Profit & Loss: 60 Months',
                 template= 'seaborn')
    fig60_SL.update_layout({'plot_bgcolor': 'lightgrey',
                 'paper_bgcolor':'grey'})

    fig3 = px.box(df_, y=yaxis_name, x=xaxis_name, color=zaxis_name,
                    labels={'grade':'Loan Grades',
                           'y':'Profit/Loss base 100',
                           'loan_status': 'Loan Status'})
    fig3.update_layout({'plot_bgcolor': 'lightgrey',
                     'paper_bgcolor':'grey',
                     'title_x': 0.5,
                     'font': {'size': 20},
                     'font_color': 'yellow'})
    print('box - attempt')
#

    return fig1, fig, line_text, fig3, fig36_SL, fig60_SL

@app.callback(
    [Output('years_lb', 'children'),
    Output('grade_lb', 'children')],
    [Input('submit_button', 'n_clicks')],
    [State('dropdown', 'value'),
    State('year_slider', 'value'),
    State('radio_l', 'value'),
    State('radio_r', 'value'),
     State('grade_items', 'value'),
     State('xaxis', 'value'),
     State('years_lb', 'children')
     ])
def update_card_title_1(n_clicks,
                        year_slider_value,
                        radio_l_value,
                        radio_r_value,
                        grade_items_value,
                        xaxis_name,
                        years_lb_children):

    out_text = 'Years: ' + str(year_slider_value[0]) + '-' + str(year_slider_value[1])
    grade_out = 'Grade: ',', '.join(grade_items_value)
    return out_text, grade_out


if __name__ == '__main__':
    app.run_server()
