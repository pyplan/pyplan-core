import os
import sys
from time import time

from pyplan_core.pyplan import Pyplan

def main():

    pyplan = Pyplan()

    t_start = time()
    filename = os.path.dirname(os.path.abspath(__file__)) + "/tests/models/Pruebas Dynamic/Pruebas Dynamic.ppl"
    pyplan.openModel(filename)

    t_model_ok = time()

    print(f'Open model: {t_model_ok-t_start}')
    value = pyplan.getResult("check_sum_all").item(0)
    print(value)
    if int(value) != 5635400435:
        print(f"\033[91m###### VALUE ERROR ###### ----->   {int(value)} != 5635400435 \033[0m" )


    t_end = time()

    print(f'Calc: {t_end-t_model_ok}')
    print(f'Total time: {t_end-t_start}')

    pyplan.closeModel()
    sys.stdout.flush()



if __name__ == '__main__':
    main()


