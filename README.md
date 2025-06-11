---
typora-root-url: ./
---

# MLProj

工程结构:

```xml
├── MLProj  
│   ├── byJupyter # 使用jupyter构建的脚本  
│   │   ├── 01-数据处理
│   │   ├── 02-数据绘图
│   │   ├── 03-预报数据评估
		└── 03-02基于模型结果进行评估_基于修正后的模型.ipynb 		 #1
		└── 03-03原始风场与修订后的rmse叠加显示.ipynb 			#2

│   ├── model_train  # 训练模型的代码
│   │   └── 04-01 LSTM.py  # 训练模型代码  
│   ├── utils  // django 项目  
│   ├── docker_commit  // docker-compose 部署的 compose files  
        ├── docker-compose.yml 
 
```



注意:

目前的预报数据目录为:

```python
forecast_path: str = r'Z:\SOURCE_MERGE_DATA\df_ws_forecast.csv'
```

实况数据为:

```python
realdata_path: str = r'Z:\SOURCE_MERGE_DATA\2024_local_df_utc_183_split.csv'
```

训练后的模型目录为:

```python
model_path: str = r'E:\05DATA\fit_model_v2_250609.h5'
```

预报与实况的归一化器存储目录为：

```python
scaler_forecast: str = r'Z:\01TRAINNING_DATA\scaler\scaler_forecast_250609.sav'
scaler_realdata: str = r'Z:\01TRAINNING_DATA\scaler\scaler_realdata_250609.sav'
```

注意实况数据最后一列由于拼接有问题，故需要去掉forecast数据的最后一列

预报与实况的`dataframe`的`shape`为: `(61,732)`

目前的训练模型参数信息如下:

```xml
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 masking (Masking)           (None, 61, 1)             0        
                                                                 
 bidirectional (Bidirectiona  (None, 61, 256)          133120    
 l)                                                                                                                               
 dropout (Dropout)           (None, 61, 256)           0         
                                                                 
 bidirectional_1 (Bidirectio  (None, 61, 128)          164352    
 nal)                                                            
                                                                 
 dropout_1 (Dropout)         (None, 61, 128)           0         
                                                                 
 dense (Dense)               (None, 61, 25)            3225      
                                                                 
=================================================================
...
Trainable params: 300,697
Non-trainable params: 0
_________________________________________________________________
```



### 一些主要脚本的说明:

（1）[03-02基于模型结果进行评估_基于修正后的模型]() :

该代码读取模型，并加载预报与实况数据，shape均为`(61, 732)`。预报与实况数据提取[:-20%]作为测试与验证数据集。并计算LSTM订正后的数据的RMSE至 `r'Z:\03TRAINNING_EVALUATION_DATA\rmse_forecast_250609.csv'`

原始风场模型的误差评估RMSE在 ` r'Z:\03TRAINNING_EVALUATION_DATA\rmse_forecast_source_250609.csv'`



(2) [03-03原始风场与修订后的rmse叠加显示]()

![004](/documents/pics/004.png)