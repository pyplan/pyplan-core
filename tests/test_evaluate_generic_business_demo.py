import os,sys,inspect
import json        

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from pyplan_core.pyplan import Pyplan

pyplan = Pyplan()


def test_openModel():
    filename = os.path.dirname(os.path.abspath(__file__)) + "/models/Generic Business Demo/Generic Business Demo.ppl"
    pyplan.openModel(filename)
    assert True , "Error on open Model" 

def test_listnodes():
    node_list = pyplan.getNodeList()
    print(node_list)
    assert len(node_list)>0, "Error on list nodes"

def test_getNode():
    node = pyplan.getNode("check_sum_all")
    assert not node is None, "Error on getnode"    

def test_evaluateNode():
    value = pyplan.getResult("check_sum_all")
    print(value)
    assert value == 4397867.429125782, "Error on evaluate node. The node result is " + str(value)

def test_closeModel():    
    pyplan.closeModel()
    assert True , "Error on release  Engine"



def test_openModel_bypass_pp_functions():
    filename = os.path.dirname(os.path.abspath(__file__)) + "/models/Generic Business Demo/Generic Business Demo pp.ppl"
    pyplan.openModel(filename)
    assert True , "Error on open Model" 

def test_evaluateNode_bypass_pp_functions():
    value = pyplan.getResult("check_sum_all")
    print(value)
    assert value == 4397867.429125782, "Error on evaluate node. The node result is " + str(value)

def test_closeModel_bypass_pp_functions():
    pyplan.closeModel()
    assert True , "Error on release  Engine"    

