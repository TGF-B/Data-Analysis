# Data-Analysis

机器学习原理比较抽象，但从案例分析开始应用，就会变得有趣多了。

## 1.支付腐败行为甄别   

- 导入[数据](https://www.kaggle.com/ealaxi/paysim1/download)    
```python    
    import pands as pd    
    import numpy as np    
    data=pd.read_csv("数据集路径”)   
```
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

-查看数据集中所有变量与**腐败与否**（isFraud）的相关性
```python
correlation=data.corr()#传入要检查相关性的数据集
print(correlation["isFraud"].sort_values(ascending=False))#将所有变量与isFraud相关性降序排列
```
> isFraud           1.000000    
> amount            0.076688    
> isFlaggedFraud    0.044109    
> step              0.031578    
> oldbalanceOrg     0.010154    
> newbalanceDest    0.000535    
> oldbalanceDest   -0.005885    
> newbalanceOrig   -0.008148
> Name: isFraud, dtype: float64

- 开始建模
  - 变量重命名（将交易类型用数字代替，为了方便后续决策树的建立）
  ```python
  data["type"]=data["type"].map({"CASH_OUT:1,"PAYMENT":2,
                                  "CASH_IN":3,"TRANSFER":4,
                                  "DEBIT":5})
  data["isFraud"]=data["isFraud"].map({0:"No Fraud",1:"Fraud"})
  ```
    - 展示一下效果
  ```python
  >    step  type    amount  ... newbalanceDest   isFraud  isFlaggedFraud    
  >     0     1     2   9839.64  ...            0.0  No Fraud               0    
  >     1     1     2   1864.28  ...            0.0  No Fraud               0
  >     2     1     4    181.00  ...            0.0     Fraud               0
  >     3     1     1    181.00  ...            0.0     Fraud               0
  >     4     1     2  11668.14  ...            0.0  No Fraud               0
