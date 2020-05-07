import os,sys,inspect
import json        

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from pyplan_core.pyplan import Pyplan

pyplan = Pyplan()

def test_clone_git_repo():
    os.system("cd tests && git clone git@bitbucket.org:novix-sa/pyplan-private-models.git")
    os.system("cd tests && cd pyplan-private-models && git fetch && git pull")

def test_openModel():
    filename = os.path.dirname(os.path.abspath(__file__)) + "/pyplan-private-models/MOT/mot.ppl"
    pyplan.openModel(filename)
    assert True , "Error on open Model" 

def test_getNode():
    node = pyplan.getNode("check_sum_all")
    assert not node is None, "Error on getnode"    

def test_evaluateNode():
    value = pyplan.getResult("check_sum_all")
    print(value)
    assert value == 47131049.92920077, f"Error on evaluate node. The node result is {value}"

def test_closeModel():    
    pyplan.closeModel()
    assert True , "Error on release  Engine"
