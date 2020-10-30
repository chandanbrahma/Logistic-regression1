
## importing the dataset
import pandas as pd
data=pd.read_csv('E:\\assignment\\logistic regression\\bank-full.csv')
data.head()
data.info()
data.describe()


##getting the unique values of the target column

data['y'].value_counts()
## so we can see that nearly 13%of the population 
data.drop(columns=['education','housing','loan','contact','day','month','campaign','pdays','previous','poutcome'],inplace=True)


##coverting the catagoricaal columns
data['job'],_ = pd.factorize(data.job)
data['marital'],_ = pd.factorize(data.marital)
data['default'],_=pd.factorize(data.default)
data['y'],_=pd.factorize(data.y)



##checking for null values
data.isnull().sum()
## we donot have any  null value so to can move to visualisation
## catagory columns=[job,marital,default,y]
## numerical column=[age,balance,duration]

##ploting catagorical column using countplot
import seaborn as sb
sb.countplot(x='job',data=data)
##so we can see different no of person in different occupation
sb.countplot(x='marital',data=data)
##maximum of the person are married
sb.countplot(x='default',data=data)
sb.countplot(x='y',data=data)
##very less person have opted for thr termdeposit

##ploting box plot for numerical variable with respect to catagorical variable

sb.boxplot(data = data,orient = "v")
sb.boxplot(x="job",y="age",data=data,palette = "hls")
sb.boxplot(x="job",y="balance",data=data,palette="hls")
sb.boxplot(x="job",y="duration",data=data,palette="hls")
## we can visualize the mean meadian as well as outlayers

sb.boxplot(x="marital",y="age",data=data,palette = "hls")
sb.boxplot(x="marital",y="balance",data=data,palette="hls")
sb.boxplot(x="marital",y="duration",data=data,palette="hls")

sb.boxplot(x='default',y="age",data=data,palette = "hls")
sb.boxplot(x="default",y="balance",data=data,palette="hls")
sb.boxplot(x="default",y="duration",data=data,palette="hls")

sb.boxplot(x="y",y="age",data=data,palette = "hls")
sb.boxplot(x="y",y="balance",data=data,palette="hls")
sb.boxplot(x="y",y="duration",data=data,palette="hls")

##model building
x=data.iloc[:,:6]
y=data.iloc[:,6]


##splliting the data into train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)



##importing the logistic regression class
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
#true positive= 7918
#true negative= 174
#false positive= 113
#false negative= 838

## precision, recall, f1-score and support from classification report
from sklearn.metrics import classification_report
print(classification_report(y_test,prediction))


