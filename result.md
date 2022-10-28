# 5-trial average scores

## CMU-MOSI
|                     |MAE |Corr|
|:--------------------|---:|---:|
|mmdatasdk(conv. feat)|.934|.667|
|mmsa(conv-BERT feat) |.889|.691|
|P/T Enc. output      |.844|.716|
|P/T Enc. mid-best    |.812|.747|
|P/T Enc. weighted    |.833|.751|

- result files and scores 
```
cmumosi/mmdatasdk_noalign/input_videoaudiotext.model_enc1x128sap4dec1_gate_dec2.train_mb16ep50adam0.0001schedspecaug.feat_layer23/result/result.tst.txt:[task: reg_regress] MAE=0.9343 Corr=0.6677
cmumosi/mmsa_noalign/input_videoaudiotext.model_enc1x128sap4dec1_gate_dec2.train_mb16ep50adam0.0001schedspecaug.feat_layer23/result/result.tst.txt:[task: reg_regress] MAE=0.8892 Corr=0.6906
cmumosi/pretrained/input_videoaudiotext.model_enc1x128sap4dec1_gate_dec2.train_mb16ep50adam0.0001schedspecaug.feat_layer23/result/result.tst.txt:[task: reg_regress] MAE=0.8440 Corr=0.7163
cmumosi/pretrained/input_videoaudiotext.model_enc1x128sap4dec1_gate_dec2.train_mb16ep50adam0.0001schedspecaug.feat_layerall/result/result.tst.txt:[task: reg_regress] MAE=0.8119 Corr=0.7465
cmumosi/pretrained/input_videoaudiotext.model_enc1x128sap4dec1_gate_dec2.train_mb16ep50adam0.0001schedspecaug.feat_layerbestmosi/result/result.tst.txt:[task: reg_regress] MAE=0.8334 Corr=0.7510
```

## CMU-MOSEI
|                     |MAE |Corr|
|:--------------------|---:|---:|
|mmdatasdk(conv. feat)|.598|.684|
|mmsa(conv-BERT feat) |.542|.748|
|P/T Enc. output      |.521|.772|
|P/T Enc. mid-best    |.507|.789|
|P/T Enc. weighted    |.511|.785|

- result files and scores 
```
cmumosei/mmdatasdk_noalign/input_videoaudiotext.model_enc1sap4dec1_gate_dec2.train_mb16ep50adam0.0001schedspecaug.feat_layer23/result/result.tst.txt:[task: reg_regress] MAE=0.5980 Corr=0.6841
cmumosei/mmsa_noalign/input_videoaudiotext.model_enc1sap4dec1_gate_dec2.train_mb16ep50adam0.0001schedspecaug.feat_layer23/result/result.tst.txt:[task: reg_regress] MAE=0.5423 Corr=0.7478
cmumosei/pretrained/input_videoaudiotext.model_enc1sap4dec1_gate_dec2.train_mb16ep50adam0.0001schedspecaug.feat_layer23/result/result.tst.txt:[task: reg_regress] MAE=0.5217 Corr=0.7721
cmumosei/pretrained/input_videoaudiotext.model_enc1sap4dec1_gate_dec2.train_mb16ep50adam0.0001schedspecaug.feat_layerall/result/result.tst.txt:[task: reg_regress] MAE=0.5073 Corr=0.7887
cmumosei/pretrained/input_videoaudiotext.model_enc1sap4dec1_gate_dec2.train_mb16ep50adam0.0001schedspecaug.feat_layerbestmosei/result/result.tst.txt:[task: reg_regress] MAE=0.5113 Corr=0.7845
```
