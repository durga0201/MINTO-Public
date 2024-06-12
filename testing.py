import minto

exp = minto.Experiment(name="trial_02")
for x in range(3):
     y = x ** 2
     with exp.run():
         exp.log_parameter("x", x)
         exp.log_result("y", y)
exp.table()