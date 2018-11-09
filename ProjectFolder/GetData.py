import pandas

data = pandas.read_csv('E0.csv')
inputdata = data.iloc[:,23:44]
inputdata['FTR'] = data['FTR']

def change_ftr(x):
    if x == "H":
        return 1
    elif x == "D":
        return 0
    return 2

inputdata['FTR'] = inputdata['FTR'].apply(change_ftr)
inputdata.to_csv('inputdata.csv', index = False)
