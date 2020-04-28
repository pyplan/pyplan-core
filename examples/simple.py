import os
from pyplan_core.pyplan import Pyplan


pyplan = Pyplan()

filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),"PyplanCoreSample.ppl")

print("Opening model...")
pyplan.openModel(filename)

print("Getting node result")
value = pyplan.getResult("total_of_cases")
print(value)

print("Closing model...")
pyplan.closeModel()




