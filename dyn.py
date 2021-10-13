import os
from time import time

from pyplan_core.pyplan import Pyplan

tests = [
    {
        "name": "Generic Business Demo",
        "filename": os.path.dirname(os.path.abspath(__file__)) + "/tests//models/Generic Business Demo/Generic Business Demo pp.ppl",
        "value": 4397867,
        "node": "check_sum_all",
        "old_time": 11
    },
    {
        "name": "Pruebas Dynamic",
        "filename": os.path.dirname(os.path.abspath(__file__)) + "/tests/models/Pruebas Dynamic/Pruebas Dynamic.ppl",
        "value": 5635400435,
        "node": "check_sum_all",
        "old_time": 54
    },
    {
        "name": "SIPME",
        "filename": os.path.dirname(os.path.abspath(__file__)) + "/tests/pyplan-private-models/SIPME/sipme.ppl",
        "node": "check_sum_all",
        "value": 70827042042,
        "old_time": 98
    },
    {
        "name": "EDP",
        "filename": os.path.dirname(os.path.abspath(__file__)) + "/tests/pyplan-private-models/MIP/PYPLAN MIP_v2021_09_08 v5.ppl",
        "node": "dis_fin_bal_controle",
        "value": 0,
        "old_time": 900
    }
]

def main():

    run_test = tests[0]

    pyplan = Pyplan()

    t_start = time()
    filename = run_test['filename']

    
    print(f'#################### running test : {run_test["name"]} ####################')
 
    pyplan.openModel(filename)

    t_model_ok = time()
    t_end = None

    print(f'Open model: {round(t_model_ok-t_start,4)}')
    try:
        value = pyplan.getResult(run_test['node'])
        print(value)
        
        if run_test['value']>0 and int(value) != run_test['value']:
            print(f"\033[91m###### VALUE ERROR ###### ----->   {int(value)} != {run_test['value']} \033[0m" )

    except Exception as ex:
        t_end = time()
        print(f'ERROR: {round(t_end-t_model_ok,4)}')
        raise(ex)

    t_end = time()
    print(f'Calc: {round(t_end-t_model_ok,4)} seg.  {round((t_end-t_model_ok)*100/run_test["old_time"],4)}%')

    print(f'\nTotal time: {round(t_end-t_start,4)}\n\n')

    pyplan.closeModel()


if __name__ == '__main__':
    main()


