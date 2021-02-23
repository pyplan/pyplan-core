import os,sys,inspect
import json

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from pyplan_core.pyplan import Pyplan

pyplan = Pyplan()

def test_openModel():
    # this is a sample .ppl file. You can use your own .ppl model file here
    filename = pyplan.sample_models.use_of_pyplan_core()
    pyplan.openModel(filename)
    assert True , "Error on open Model" 

def test_createDashApp():
    dash_id=123
    code="""
    import dash
    import dash_core_components as dcc
    import dash_html_components as html
    from dash.dependencies import Input, Output
    import plotly.express as px
    import pandas as pd

    app = dash.Dash()


    app.layout = html.Div([
        html.H2("Simple plotly"),
        html.Div(dcc.Slider(id='my-input', min=10, max=100, value=50), style={'width': '500px'}),
        dcc.Graph(id='simple-fig'),
    ])


    @app.callback(
        Output(component_id='simple-fig', component_property='figure'),
        Input(component_id='my-input', component_property='value')
    )
    def update_output_div(input_value):
        
        df = pd.DataFrame({
                "x": [1,2,1,2],
                "y": [1,2,3,4],
                "customdata": [1,2,3,4],
                "fruit": ["apple", "apple", "orange", "orange"]
            })

        fig = px.scatter(df, x="x", y="y", color="fruit", custom_data=["customdata"])
        fig.update_traces(marker_size=int(input_value))

        return fig
    """ 
    params = {
        'path': "/",
        'data': None,
        'method': "GET",
        'content_type': 'text/html'
    }

    res = pyplan.model.createDashApp(dash_id, code, params)
    print("\n")
    print(res)
    assert len(res)==1675, "Error on createDashApp"    

def test_dispatch_dependencies():
    dash_id=123
    params = {
        'path': "/_dash-dependencies",
        'data': None,
        'method': "GET",
        'content_type': 'text/json'
    }
    res = pyplan.model.dispatchDashApp(dash_id, params)
    print("\n")
    print(res)
    assert len(res)==148, "Error on dispatch_dependencies"    

def test_dispatch_update():
    dash_id=123
    params = {
        'path':'/_dash-update-component',
        'data':'{"output":"simple-fig.figure","outputs":{"id":"simple-fig","property":"figure"},"inputs":[{"id":"my-input","property":"value","value":34}],"changedPropIds":["my-input.value"]}'.encode(),
        'method':'POST',
        'content_type':'application/json'
    }
    
    res = pyplan.model.dispatchDashApp(dash_id, params)
    print("\n")
    print(res)
    assert len(res)==8366, "Error on dispatch_update"    



def test_closeModel():    
    pyplan.closeModel()
    assert True , "Error on release  Engine"
