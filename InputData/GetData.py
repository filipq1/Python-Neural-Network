import pandas

data = pandas.read_csv('E0.csv')
inputdata = data.iloc[:,23:44]
inputdata['FTR'] = data['FTR']
inputdata['FTR'] = inputdata['FTR'].astype('category')
inputdata['FTR'] = inputdata['FTR'].cat.codes
inputdata.to_csv('inputdata.csv', index = False)