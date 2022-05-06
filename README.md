
机器学习原理比较抽象，但从案例分析开始应用，就会变得有趣多了。

## 交易欺诈行为识别  

由于极高的便捷性，网上交易已经成为当今社会最主流的交易方式，但也正是这个”优点”，交易欺诈类的新闻总是不绝于耳。为了提高交易系统的安全性，我们可以基于过往交易的数据建立一套欺诈行为识别模型，之后若有符合欺诈特征的交易动作发生，我们可以提前采取行动，减少损失。

- 导入[数据](https://www.kaggle.com/ealaxi/paysim1/download)    

```python    
   import pands as pd    
   import numpy as np    
   data=pd.read_csv("数据集路径”)   
```
- 数据预处理
    
    - 看一下数据集的头部
    ```python    
       data.head()
    ```
    >    step      type    amount  ... newbalanceDest  isFraud  isFlaggedFraud    
    0     1   PAYMENT   9839.64  ...            0.0        0               0    
    1     1   PAYMENT   1864.28  ...            0.0        0               0    
    2     1  TRANSFER    181.00  ...            0.0        1               0    
    3     1  CASH_OUT    181.00  ...            0.0        1               0    
    4     1   PAYMENT  11668.14  ...            0.0        0               0        
    
   - 定义:   
    
    > 1.step:耗时一个小时以上的步骤        
    > 2.type:网上交易的类型        
    > 3.amount：网上交易的总额      
    > 4.nameOrig：交易发起人       
    > 5.oldbalanceOrg:发起人交易前余额      
    > 6.newbalanceOrg:发起人交易后余额      
    > 7.nameDest:交易接收人      
    > 8.oldbalanceDest:接收人交易前余额     
    > 9.newbalanceDest:接收人交易后余额     
    > 10.isFraud:欺诈交易
    - 检查数据集中是否有空白项     
    ```python
       print(data.isnull().sum())    
    ```
    - 按类别求总和
    ```python
       print(data.type.value_counts())
    ```
    > CASH_OUT    2237500    
    > PAYMENT     2151495   
    > CASH_IN     1399284    
    > TRANSFER     532909    
    > DEBIT         41432    
    > Name: type, dtype: int64

    - 设置关键参数
    ```python
       type=data['type'].value_counts() #引用上一步得到的交易类型和每种类型的交易总额
       transactions=type.index #参数重命名
       quantity=type.values  #参数重命名                
     ```  
- 绘制交易类型分布扇形图
```python
import plotly.express as px
figure=px.pie(data,
           values=quantity,
           names=transactions,hole=0,5,
           title="Distribution of Transaction Type")
figure.show()
 ```
 ![得到图形]（C:\Users\Administrator\Desktop\python learning\pandas learning\Payment Fraud Detection/newplot.png）

  
- 变量重命名（将各种属性都用数字代替，方便后续决策树的建立）    
```python
   data["type"]=data["type"].map({"CASH_OUT:1,"PAYMENT":2,
                                  "CASH_IN":3,"TRANSFER":4,
                                  "DEBIT":5}) #提现：1，支付，入账，转账，
   data["isFraud"]=data["isFraud"].map({0:"No Fraud",1:"Fraud"})
 ```    
 - 查看数据集中所有变量（属性）与**欺诈与否**（isFraud）的相关性
 ```python
    correlation=data.corr()#传入要检查相关性的数据集
    print(correlation["isFraud"].sort_values(ascending=False))#将所有变量(属性）与欺诈与否（isFraud）相关性降序排列
 ```        
返回：
    > isFraud           1.000000    
    > amount            0.076688    
    > isFlaggedFraud    0.044109    
    > step              0.031578    
    > oldbalanceOrg     0.010154    
    > newbalanceDest    0.000535    
    > oldbalanceDest   -0.005885    
    > newbalanceOrig   -0.008148        
    > Name: isFraud, dtype: float64
显然，从相关性来看，交易额（amount），是否被标注过
- 数据集划分    
```python
   from sklearn.model_selection import train_test_split
   x=np.array(data[["type","amount","oldbalanceOrg","newbalanceOrig"]])#选出四个与isFraud相关性最强的变量（属性）构成决策树的自变量（划分属性）
   y=np.array(data[["isFraud"]])
   xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.1,random_state=42)#90%的数据做训练集，10%的数据做测试集
```
- 建立决策树分类器

```python
   from sklearn.tree import DecisionTreeClassifier
   model=DecisionTreeClassifier()
   model.fit(xtrain,ytrain)#用90%的数据集训练决策树
   print(model.score(xtest,ytest)#用10%的数据集验证决策树，并返回拟合度
```
 很顺利地返回：    
 > 0.9997391011878755  

说明决策树模型构建地很成功。
 
 - 预测    

现在我们输入一组交易信息，让模型来预测这组交易是否存在欺诈行为。

```python
   features=np.array([[4,9000.60,9000.60,0.0]])
   print(model.predict(features))
```   
返回：

> ['Fraud']     

说明这组交易很可能存在欺诈行为，需要针对性地采取行动。
