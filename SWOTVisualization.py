import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import os,sys,json
import pandas as pd
import numpy as np

def ploteCDFs(dfs,variable,names):
    fig = go.Figure()
    
    i=0
    for df in dfs:
        xval=df[variable].sort_values()
        yval=np.linspace(0,1,len(xval))
        fig.add_trace(go.Scatter(x=xval,y=yval,name=names[i]))
        i+=1

    fig.update_layout(
        xaxis_range=[0,1.5],
        xaxis_title="67th percentile ε",
        yaxis_title='empirical CDF',
        width=800,
        height=400
    )
    
    fig.show()


def plotbarn(dfs,expnames):
    
    dfns=[]
    i=0
    for df in dfs:
        dfns.append(pd.DataFrame(data={'n':[len(df)],'name':[expnames[i]],'type':['reach'] }))
        dfns.append(pd.DataFrame(data={'n':[df['n'].sum()],'name':[expnames[i]],'type':['times'] }))
        i+=1
        
    dfn=pd.concat(dfns)
        
    # return dfn
    
    fig=px.bar(dfn,x='name',y='n',facet_col='type',barmode='group')
    fig.update_yaxes(matches=None)
    fig.update_yaxes(showticklabels=True, col=2) # assuming second facet
    
    
    fig.show()