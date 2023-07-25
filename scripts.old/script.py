import subprocess


inducing = [20, 50, 100]
prior_stds = [0.01, 0.1]
weight_decay = 0.001

a = "c:/Users/Ludvins/Documents/DeepVIP/.venv_laplace/Scripts/python.exe "

ella_reg = "c:/Users/Ludvins/Documents/DeepVIP/scripts/ella_regression.py "
valla_reg = "c:/Users/Ludvins/Documents/DeepVIP/scripts/valla_regression.py "

valla = "c:/Users/Ludvins/Documents/DeepVIP/scripts/valla_multiclass.py "
ella = "c:/Users/Ludvins/Documents/DeepVIP/scripts/ella_multiclass.py "

valla_ood = "c:/Users/Ludvins/Documents/DeepVIP/scripts/ella_multiclass_ood.py "
valla_ood2 = "c:/Users/Ludvins/Documents/DeepVIP/scripts/valla_multiclass_ood2.py "
ella_ood = "c:/Users/Ludvins/Documents/DeepVIP/scripts/ella_multiclass_ood.py "

for i in prior_stds:
    for induci in inducing:
        subprocess.run(
            a
            + valla_ood
            + "--dataset MNIST_OOD --MAP_iterations 100000 --iterations 20000 --split 0 --num_inducing {} --weight_decay {} --prior_std {}".format(
                induci, weight_decay, i
            )
        )
        subprocess.run(
            a
            + ella_ood
            + "--dataset MNIST_OOD --MAP_iterations 100000 --iterations 20000 --split 0 --num_inducing {} --weight_decay {} --prior_std {}".format(
                induci, weight_decay, i
            )
        )
        # subprocess.run(a + valla_ood2 + "--dataset CIFAR10_OOD --MAP_iterations 100000 --iterations 20000 --split 0 --num_inducing {} --weight_decay {} --prior_std {}".format(induci, weight_decay, i))


""" for i in inducing:
    for j in prior_stds:
        subprocess.run(a + ella + "--dataset MNIST_OOD --MAP_iterations 50000 --iterations 50000 --split 0 --num_inducing {} --prior_std {} ".format(i, j))
        

 """
