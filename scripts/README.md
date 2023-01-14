# Script
训练和测试模板脚本都在`scripts`文件夹下，运行如下：
```bash
#./scripts/tmp_dist_train.sh ${GPU_ID} ${GPU_NUM} [optional arguments]
./scripts/tmp_dist_train.sh 0,1 2 --cfg_file config/model_configs/persformer.yaml --batch_size=8 --nepochs=40 --exp=PersFormer
./scripts/qzj/dist_train.sh 0,1,2,3 4 --cfg_file config/model_configs/persformer.yaml --batch_size=8 --nepochs=20 --exp=PersFormer_e100
```
```bash
#./scripts/dist_eval_case.sh ${GPU_ID} ${GPU_NUM} [optional arguments]
./scripts/dist_eval_case.sh 0 1 --cfg_file config/model_configs/persformer.yaml --batch_size=10
```
如果有自己的喜好，可以像我一样新建个自己的文件夹。
Note: 如果shell脚本没有执行的权限，可以使用`chmod +x scripts/tmp_dist_train.sh`来添加执行权限。