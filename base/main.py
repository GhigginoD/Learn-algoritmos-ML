import numpy as np
import pandas as pd


def loadDataSet(filename:str):
  print('Carregando a base de dados...')
  baseDeDados = pd.read_csv(f'{filename}.csv', delimiter=';')
  x = baseDeDados.iloc[:,:-1].values
  y = baseDeDados.iloc[:,-1].values

  return x, y

def fillMissingData(x):
    print("Preenchendo dados que estão faltando...")
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    x[:,1:] = imputer.fit_transform(x[:,1:])
    print(x)
    return x

def computeCategorization(x):
  print("Computando rotulação...")
  from sklearn.preprocessing import LabelEncoder
  labelEncoder_x = LabelEncoder()
  x[:,0] = labelEncoder_x.fit_transform(x[:,0])
  print('x:',x)

  d = pd.get_dummies(x[:,0])
  x = x[:,1:]
  x = np.insert(x, 0 ,d.values, axis=1)
  print('hot encoding!')
  return x

def splitTrainTestSets(x, y):
  print('Separando conjunto de testes e treino...')
  from sklearn.model_selection import train_test_split
  xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size=0.8)
  print('ok!')
  return xtrain, xtest, ytrain, ytest

def computeLinearRegression(xtrain, ytrain, xtest, ytest):
  import matplotlib.pyplot as plt
  from sklearn.linear_model import LinearRegression
  print('Computando Regressão Linear...')
  regressor = LinearRegression()
  regressor.fit(xtrain, ytrain)
  ypredict = regressor.predict(xtest)

  print("ok!")
  plt.scatter(xtest[:,-1], ytest, color = 'red')
  plt.plot(xtest[:,-1], ypredict, color='blue')
  plt.title("Inscritos x Visualizações")
  plt.xlabel("Inscritos")
  plt.ylabel("Visualizações")
  plt.show()

def runLinearRegressionExample():
    X, y = loadDataSet("base/dataSet")
    X = fillMissingData(X)
    X = computeCategorization(X)
    XTrain, XTest, yTrain, yTest = splitTrainTestSets(X, y)
    computeLinearRegression(XTrain, yTrain, XTest, yTest)

if __name__ == "__main__":
    runLinearRegressionExample()