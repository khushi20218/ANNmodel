import joblib
import pandas
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
dataset  =  pandas.read_csv('train.csv')
gender = dataset['Sex']
age = dataset['Age']
def lw(cols):
    age = cols[0]
    Pclass = cols[1]
    if pandas.isnull(age):
        if Pclass == 1:
            return 38
        elif Pclass == 2:
            return 30
        elif Pclass == 3:
            return 25
        else:
            return 30
    else:
        return age
    
dataset['Age'] = dataset[['Age', 'Pclass']].apply(lw , axis=1)
dataset.drop('Cabin', axis=1, inplace=True)
fare = dataset['Fare']
y = dataset['Survived']
X = dataset[ ['Pclass','Sex', 'Age', 'SibSp', 'Parch' , 'Embarked' ]]
sex = dataset['Sex']
sex = pandas.get_dummies(sex, drop_first=True )
pclass = dataset['Pclass']
pclass = pandas.get_dummies(pclass, drop_first=True)
sibsp = dataset['SibSp']
sibsp = pandas.get_dummies(sibsp, drop_first=True)
parch = dataset['Parch']
parch = pandas.get_dummies(parch, drop_first=True)
embarked = dataset['Embarked']
embarked = pandas.get_dummies(embarked, drop_first=True)
age = dataset[ 'Age']
X = pandas.concat([age, embarked, parch, sibsp, pclass, sex] ,  axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
model = LogisticRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
a=(classification_report(y_test, y_pred))
b=a.split('\n')
c=b[-2]
d=c.split()
e=d[2]
f=float(e)
acc=f*100
print(acc)
joblib.dump(model,'ml.pk1')







