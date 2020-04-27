import os,sys,inspect
import json        
from pytest import approx

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

#from app.classes.CalcEngine import CalcEngine
from pyplan import Pyplan

pyplan = Pyplan()


def test_openModel():
    #global pyplan
    filename = os.path.dirname(os.path.abspath(__file__)) + "/models/Generic Business Demo/Generic Business Demo.ppl"
    pyplan.openModel(filename)
    assert True , "Error on open Model" 

def test_evaluateNode():
    res = pyplan.getResult("check_sum_all")
    obj = json.loads(res)
    value = obj["result"]
    assert value == approx(272231819453.77332) , "Error on evaluate node. The node result is " + str(value)

def test_closeModel():    
    pyplan.closeModel()
    assert True , "Error on release  Engine"
