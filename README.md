# %% [code] {"execution":{"iopub.status.busy":"2025-10-26T10:31:46.615799Z","iopub.execute_input":"2025-10-26T10:31:46.616584Z","iopub.status.idle":"2025-10-26T10:31:46.625815Z","shell.execute_reply.started":"2025-10-26T10:31:46.616554Z","shell.execute_reply":"2025-10-26T10:31:46.624839Z"},"jupyter":{"outputs_hidden":false}}
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot as plt
import os
from sklearn.metrics import r2_score

dataset=pd.read_csv('/kaggle/input/global-crocodile-species-dataset/crocodile_dataset.csv')
df=pd.DataFrame(dataset)
df.duplicated().sum()
df.dropna()
df.drop(['Common Name'],axis=1)

print(df.columns)

x=df['Observed Length (m)']
y=df['Observed Weight (kg)']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

rf_model=RandomForestRegressor(n_estimators=100,random_state=42)

x_train=x_train.values.reshape(-1,1)

rf_model.fit(x_train,y_train)

print("rf model trained successfully")

x_test=x_test.values.reshape(-1,1)
predictions=rf_model.predict(x_test)
score=r2_score(y_test,predictions)

print(f"Model R-squared score: {score:.4f}")

plt.figure(figsize=(10,6))
x_axis=x_train
y_axis=x_test
plt.plot(y_test,predictions,color='red',lw='2')
