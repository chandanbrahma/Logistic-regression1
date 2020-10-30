## importing the dataset
import pandas as pd
data=pd.read_csv('E:\\assignment\\logistic regression\\creditcard.csv')
data.head()
data.info()
data.describe()


##getting the unique values of the target column

data['card'].value_counts()
##so for 77% of the cases card is accepted
data.drop(columns=['owner','dependents','months'],inplace=True)


##there are 2 catagorical columns which needs to be converted to integer type
data.card,_= pd.factorize(data.card)
data.selfemp,_=pd.factorize(data.selfemp)


data.isnull().sum()
##as we do not have any nulll values in our datset, we are good to go with the visualisation


##catagorical column= card, selfemp
##integer column= reports,age, income,share,expenditure,majorcards,active

#counnt plot for catagorical columns
import seaborn as sb
sb.countplot(x='card',data=data)
##more no aplications are accepted
sb.countplot(x='selfemp',data=data)
##less no are self employeed

##box plot to visualize the integer column with all tthe catagorical values
sb.boxplot(data = data,orient = "v")
sb.boxplot(x="card",y="age",data=data,palette = "hls")
sb.boxplot(x="card",y="reports",data=data,palette="hls")
sb.boxplot(x="card",y="income",data=data,palette="hls")
sb.boxplot(x="card",y="share",data=data,palette="hls")
sb.boxplot(x="card",y="expenditure",data=data,palette="hls")
sb.boxplot(x="card",y="majorcards",data=data,palette="hls")
sb.boxplot(x="card",y="active",data=data,palette="hls")

##we can visualize the mean and outlayers.also we can visualize different no of cards issuied with respect to different parameters
sb.boxplot(data = data,orient = "v")
sb.boxplot(x="selfemp",y="age",data=data,palette = "hls")
sb.boxplot(x="selfemp",y="reports",data=data,palette="hls")
sb.boxplot(x="selfemp",y="income",data=data,palette="hls")
sb.boxplot(x="selfemp",y="share",data=data,palette="hls")
sb.boxplot(x="selfemp",y="expenditure",data=data,palette="hls")
sb.boxplot(x="selfemp",y="majorcards",data=data,palette="hls")
sb.boxplot(x="selfemp",y="active",data=data,palette="hls")


#choosing x and y for modelbuilding
x=data.iloc[:,1:]
y=data.iloc[:,0]


##splliting the data into train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train, y_train)


##predicting the data
prediction = logreg.predict(x_test)


##using connfusion matrix to predict the accuracy
from sklearn.metrics import confusion_matrix
conf_Matrix = confusion_matrix(y_test, prediction)
print(conf_Matrix)
##so from our confusion matrix we conclude that
#true positive= 196
#true negative= 63
#false positive= 5
#false negative= 0

## precision, recall, f1-score and support from classification report
from sklearn.metrics import classification_report
print(classification_report(y_test,prediction))
