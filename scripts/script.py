import subprocess

inducing = [5, 10, 20, 30, 40, 50]

a = "c:/Users/Ludvins/Documents/DeepVIP/.venv_laplace/Scripts/python.exe "
b = "c:/Users/Ludvins/Documents/DeepVIP/scripts/ella_multiclass2D.py "
d = "c:/Users/Ludvins/Documents/DeepVIP/scripts/valla_aram_multiclass2D.py "

for i in inducing:
    subprocess.run(a + b + "--dataset Spiral3 --MAP_iterations 10000 --iterations 20000 --split 0 --num_inducing {} --prior_std 0.1".format(i))
