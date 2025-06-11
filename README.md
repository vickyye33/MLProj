---
typora-root-url: ./
---

# MLProj

### 1 工程结构说明:

```xml
├── MLProj  
│   ├── byJupyter # 使用jupyter构建的脚本  
│   │   ├── 01-数据处理 									 # 3
		└──	01-提取整年的风场数据_imp
		└── 02-将fub北京时转换为utc时间_imp 					  # 4
│   │   ├── 02-数据绘图
│   │   ├── 03-预报数据评估
		└── 03-02基于模型结果进行评估_基于修正后的模型.ipynb 		#1
		└── 03-03原始风场与修订后的rmse叠加显示.ipynb 		    #2
│   ├── model_train  # 训练模型的代码
│   │   └── 04-01 LSTM.py  # 训练模型代码  
│   ├── utils  # 工具目录  
		└── common.py  # 公共工具类
			└── def get_realdata_df # 读取 fub 实况数据并返回 dataframe
```

* num 作为 2 对应代码的索引

注意:

* `Z:\ `为家中NAS 目录

目前的

预报数据目录为:

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



### 2 一些主要脚本的说明:

（1）[03-02基于模型结果进行评估_基于修正后的模型]() :

该代码读取模型，并加载预报与实况数据，shape均为`(61, 732)`。预报与实况数据提取[:-20%]作为测试与验证数据集。并计算LSTM订正后的数据的RMSE至 `r'Z:\03TRAINNING_EVALUATION_DATA\rmse_forecast_250609.csv'`

原始风场模型的误差评估RMSE在 ` r'Z:\03TRAINNING_EVALUATION_DATA\rmse_forecast_source_250609.csv'`



(2) [03-03原始风场与修订后的rmse叠加显示]()

![004](/documents/pics/004.png)

(3) [01-提取整年的风场数据_imp]()  

基于指定经纬度提取对应的风场时序数据，并生成 `shape:(72,732)`的数据。

目前已存储至`forecast_path: str = r'Z:\SOURCE_MERGE_DATA\df_ws_forecast.csv'`不需要多次执行。

(4) [02-将fub北京时转换为utc时间_imp]()

`GRAPES`风场数据时间步长为 3h , 预报时次为 61 。每组总计需要取 3*61=183 个实况。从fub实况数据中读取 step=1, len=183 个时次的实况数据。

```
read_file_path: str = r'Z:/FUB/MF01001/2024_local.csv'
out_put_file_path: str = r'Z:/FUB/MF01001/2024_local_df_utc_183.csv'
```



### 3 后续计划:

##### （1）使用以上方式评估 `fub_b` 以及`fub_c `，测试当前模式的适用性。

##### （2）改进model

##### （3）找到导致`fit_model_v2_250609`模型`index=0`时的误差较原始风场预报显著增加的成因。