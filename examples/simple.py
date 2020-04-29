from pyplan_core.pyplan import Pyplan

pyplan = Pyplan()

# this is a sample .ppl file. You can use your own .ppl model file here
filename = pyplan.sample_models.use_of_pyplan_core()

print("Opening model...")
pyplan.openModel(filename)

print("Getting node result...")
df = pyplan.getResult("total_of_cases")

print("Print top 10 cases...")
df = df.sort_values(["Total"], ascending=False)
print(df.head(10))

print("Closing model...")
pyplan.closeModel()