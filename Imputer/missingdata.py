

def main():
  import numpy as np
  import pandas as pd
  baseDeDados = pd.read_csv('channelyoutubers.csv', delimiter=',', encoding='windows-1252')
  X = baseDeDados.iloc[:,:].values

  from sklearn.impute import SimpleImputer

  imputer = SimpleImputer(missing_values=np.nan, strategy='median')
  imputer = imputer.fit(X[:,1:3])
  X = imputer.transform(X[:,1:3]).astype(str)
  X = np.insert(X, 0, baseDeDados.iloc[:,0].values, axis=1)

  print(X)
if __name__ == "__main__":
    main()