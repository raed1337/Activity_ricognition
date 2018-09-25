
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.preprocessing import LabelEncoder

import seaborn as sns


def importDataSegemntation():
    df_data = pd.read_csv("./ConfLongDemo_JSI.csv",header=None)
    return df_data

importDataSegemntation()

data=importDataSegemntation()
data.columns=['Sequence Name','tagID','Time stamp','Date','x coordinate','y coordinate','z coordinate','activity']
#data.columns=['Subject id','sequence id','x coordinate','y coordinate','z coordinate','sensor id','activity','instane date']
print(data.head())

# process columns, apply object > numeric
# process columns, apply LabelEncoder to categorical features
for c in data.columns:
    if data[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(data[c].values))
        data[c] = lbl.transform(list(data[c].values))
print(data.head())


f, ax = plt.subplots(figsize=(11, 15))

#ax.set_axis_bgcolor('#fafafa')
ax.set(xlim=(-.05, 50))
plt.ylabel('Dependent Variables')
plt.title("Box Plot of Pre-Processed Data Set")
ax = sns.boxplot(data=data,
                 orient='h',
                 palette='Set2')
plt.show()


# sns.set(style="ticks", color_codes=True)
# g = sns.pairplot(train, hue='Dx')


def plot_corr(df, size=10):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns);
    plt.yticks(range(len(corr.columns)), corr.columns);
    plt.show()



plot_corr(data)

new_col= data.groupby('activity').mean()
print(new_col.head())
train=data.fillna(0)

cols = ['x coordinate','y coordinate','z coordinate','activity']

sns.pairplot(train[cols],
             x_vars = cols,
             y_vars = cols,
             hue = 'activity',
             )
plt.show()

sns.pairplot(train[cols],
             x_vars = cols,
             y_vars = cols,
             kind='reg'
             )
plt.show()