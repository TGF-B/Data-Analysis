#数据导入和预处理
import pands as pd    
import numpy as np    
    data=pd.read_csv("数据集路径”)
    print(data.type.value_counts())
    type=data['type'].value_counts() #引用上一步得到的交易类型和每种类型的交易总额
    transactions=type.index #参数重命名
    quantity=type.values  #参数重命名

#绘图：
import plotly.express as px
    figure=px.pie(data,
    values=quantity,
    names=transactions,hole=0,5,
    title="Distribution of Transaction Type")
    figure.show()
#数据再次预处理                    
    data["type"]=data["type"].map({"CASH_OUT:1,"PAYMENT":2,
                                      "CASH_IN":3,"TRANSFER":4,
                                      "DEBIT":5}) #提现：1，支付，入账，转账，
    data["isFraud"]=data["isFraud"].map({0:"No Fraud",1:"Fraud"})

    correlation=data.corr()#传入要检查相关性的数据集
    print(correlation["isFraud"].sort_values(ascending=False))#将所有变量(属性）与欺诈与否（isFraud）相关性降序排列

#划分数据集
from sklearn.model_selection import train_test_split
    x=np.array(data[["type","amount","oldbalanceOrg","newbalanceOrig"]])#选出四个与isFraud相关性最强的变量（属性）构成决策树的自变量（划分属性）
    y=np.array(data[["isFraud"]])
    xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.1,random_state=42)#90%的数据做训练集，10%的数据做测试集
from sklearn.tree import DecisionTreeClassifier
    model=DecisionTreeClassifier()
    model.fit(xtrain,ytrain)#用90%的数据集训练决策树
    print(model.score(xtest,ytest)#用10%的数据集验证决策树，并返回拟合度


#输入数据，应用此模型
features=np.array([[4,9000.60,9000.60,0.0]])
print(model.predict(features))
