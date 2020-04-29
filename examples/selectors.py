import os
from pyplan_core.pyplan import Pyplan

pyplan = Pyplan()

# this is a sample .ppl file. You can use your own .ppl model file here
filename = pyplan.sample_models.use_of_pyplan_core()

print("Opening model...")
pyplan.openModel(filename)

print("Get selector node")
selector = pyplan.getResult(node_id="country_selector")

print("Set selector to US")
us_position = selector.options.index("US")
pyplan.setSelectorValue(node_id="country_selector",value=us_position)

print("Get US cases")
df = pyplan.getResult("total_of_cases")
print(df)


print("Set selector to China")
china_position = selector.options.index("China")
pyplan.setSelectorValue(node_id="country_selector",value=china_position)

print("Get China cases")
df = pyplan.getResult(node_id="total_of_cases")
print(df)

print("Closing model...")
pyplan.closeModel()




