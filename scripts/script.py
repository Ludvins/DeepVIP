import subprocess

seeds = [0, 1, 2, 3, 4]
inducing = [5, 10, 20, 30, 40]

a = "c:/Users/Ludvins/Documents/DeepVIP/.venv_laplace/Scripts/python.exe "
b = "c:/Users/Ludvins/Documents/DeepVIP/scripts/ella_binary2D.py "
d = "c:/Users/Ludvins/Documents/DeepVIP/scripts/ella_regression_kl.py "
c = "c:/Users/Ludvins/Documents/DeepVIP/scripts/valla_binary2D.py "
for s in seeds:
    for i in inducing:
        print(s, i)
        subprocess.run(a + d + "--dataset synthetic2 --MAP_iterations 10000 --iterations 20000 --split 0 --freeze_ll --num_inducing {} --seed {}".format(i, s))
