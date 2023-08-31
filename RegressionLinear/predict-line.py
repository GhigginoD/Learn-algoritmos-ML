import pandas as pd

base_de_dados = pd.read_csv('../data/exp-payment.csv', delimiter=',')
salarios = base_de_dados[:,-1].values 
print(salarios)
