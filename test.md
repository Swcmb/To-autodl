# 实验报告

- 生成时间：2025-10-13 22:55:50
- 公共参数：`python main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type normal --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5`

## 编码器类型

### csglmd

`python main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type normal --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --encoder_type csglmd --run_name csglmd`

### mgacmda

`python main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type normal --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --encoder_type mgacmda --run_name mgacmda`

### gat

`python main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type normal --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --encoder_type gat --run_name gat`

### gt

`python main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type normal --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --encoder_type gt --run_name gt`

### gat_gt_serial

`python main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type normal --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --encoder_type gat_gt_serial --run_name gat_gt_serial`

### gat_gt_parallel

`python main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type normal --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --encoder_type gat_gt_parallel --run_name gat_gt_parallel`

### 实验结果 - 编码器类型

| Run | Encoder | Fusion | AUC | ACC | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| csglmd | csglmd | - | - | - | - | - | - |
| mgacmda | mgacmda | - | - | - | - | - | - |
| gat | gat | - | - | - | - | - | - |
| gt | gt | - | - | - | - | - | - |
| gat_gt_serial | gat_gt_serial | - | - | - | - | - | - |
| gat_gt_parallel | gat_gt_parallel | - | - | - | - | - | - |

## 图融合类型

### basic

`python main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type normal --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --fusion_type basic --run_name basic`

### dot

`python main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type normal --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --fusion_type dot --run_name dot`

### additive

`python main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type normal --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --fusion_type additive --run_name additive`

### self_attn

`python main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type normal --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --fusion_type self_attn --run_name self_attn`

### gat_fusion

`python main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type normal --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --fusion_type gat_fusion --run_name gat_fusion`

### gt_fusion

`python main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type normal --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --fusion_type gt_fusion --run_name gt_fusion`

### 实验结果 - 图融合类型

| Run | Encoder | Fusion | AUC | ACC | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| basic | - | basic | - | - | - | - | - |
| dot | - | dot | - | - | - | - | - |
| additive | - | additive | - | - | - | - | - |
| self_attn | - | self_attn | - | - | - | - | - |
| gat_fusion | - | gat_fusion | - | - | - | - | - |
| gt_fusion | - | gt_fusion | - | - | - | - | - |

## 编码器类型+图融合类型（交叉实验）

### csglmd 和 basic

`python main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type normal --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --encoder_type csglmd --fusion_type basic --run_name csglmd-basic`

### csglmd 和 dot

`python main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type normal --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --encoder_type csglmd --fusion_type dot --run_name csglmd-dot`

### csglmd 和 additive

`python main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type normal --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --encoder_type csglmd --fusion_type additive --run_name csglmd-additive`

### csglmd 和 self_attn

`python main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type normal --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --encoder_type csglmd --fusion_type self_attn --run_name csglmd-self_attn`

### csglmd 和 gat_fusion

`python main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type normal --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --encoder_type csglmd --fusion_type gat_fusion --run_name csglmd-gat_fusion`

### csglmd 和 gt_fusion

`python main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type normal --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --encoder_type csglmd --fusion_type gt_fusion --run_name csglmd-gt_fusion`

### mgacmda 和 basic

`python main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type normal --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --encoder_type mgacmda --fusion_type basic --run_name mgacmda-basic`

### mgacmda 和 dot

`python main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type normal --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --encoder_type mgacmda --fusion_type dot --run_name mgacmda-dot`

### mgacmda 和 additive

`python main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type normal --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --encoder_type mgacmda --fusion_type additive --run_name mgacmda-additive`

### mgacmda 和 self_attn

`python main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type normal --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --encoder_type mgacmda --fusion_type self_attn --run_name mgacmda-self_attn`

### mgacmda 和 gat_fusion

`python main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type normal --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --encoder_type mgacmda --fusion_type gat_fusion --run_name mgacmda-gat_fusion`

### mgacmda 和 gt_fusion

`python main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type normal --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --encoder_type mgacmda --fusion_type gt_fusion --run_name mgacmda-gt_fusion`

### gat 和 basic

`python main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type normal --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --encoder_type gat --fusion_type basic --run_name gat-basic`

### gat 和 dot

`python main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type normal --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --encoder_type gat --fusion_type dot --run_name gat-dot`

### gat 和 additive

`python main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type normal --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --encoder_type gat --fusion_type additive --run_name gat-additive`

### gat 和 self_attn

`python main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type normal --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --encoder_type gat --fusion_type self_attn --run_name gat-self_attn`

### gat 和 gat_fusion

`python main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type normal --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --encoder_type gat --fusion_type gat_fusion --run_name gat-gat_fusion`

### gat 和 gt_fusion

`python main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type normal --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --encoder_type gat --fusion_type gt_fusion --run_name gat-gt_fusion`

### gt 和 basic

`python main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type normal --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --encoder_type gt --fusion_type basic --run_name gt-basic`

### gt 和 dot

`python main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type normal --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --encoder_type gt --fusion_type dot --run_name gt-dot`

### gt 和 additive

`python main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type normal --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --encoder_type gt --fusion_type additive --run_name gt-additive`

### gt 和 self_attn

`python main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type normal --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --encoder_type gt --fusion_type self_attn --run_name gt-self_attn`

### gt 和 gat_fusion

`python main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type normal --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --encoder_type gt --fusion_type gat_fusion --run_name gt-gat_fusion`

### gt 和 gt_fusion

`python main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type normal --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --encoder_type gt --fusion_type gt_fusion --run_name gt-gt_fusion`

### gat_gt_serial 和 basic

`python main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type normal --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --encoder_type gat_gt_serial --fusion_type basic --run_name gat_gt_serial-basic`

### gat_gt_serial 和 dot

`python main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type normal --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --encoder_type gat_gt_serial --fusion_type dot --run_name gat_gt_serial-dot`

### gat_gt_serial 和 additive

`python main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type normal --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --encoder_type gat_gt_serial --fusion_type additive --run_name gat_gt_serial-additive`

### gat_gt_serial 和 self_attn

`python main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type normal --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --encoder_type gat_gt_serial --fusion_type self_attn --run_name gat_gt_serial-self_attn`

### gat_gt_serial 和 gat_fusion

`python main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type normal --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --encoder_type gat_gt_serial --fusion_type gat_fusion --run_name gat_gt_serial-gat_fusion`

### gat_gt_serial 和 gt_fusion

`python main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type normal --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --encoder_type gat_gt_serial --fusion_type gt_fusion --run_name gat_gt_serial-gt_fusion`

### gat_gt_parallel 和 basic

`python main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type normal --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --encoder_type gat_gt_parallel --fusion_type basic --run_name gat_gt_parallel-basic`

### gat_gt_parallel 和 dot

`python main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type normal --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --encoder_type gat_gt_parallel --fusion_type dot --run_name gat_gt_parallel-dot`

### gat_gt_parallel 和 additive

`python main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type normal --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --encoder_type gat_gt_parallel --fusion_type additive --run_name gat_gt_parallel-additive`

### gat_gt_parallel 和 self_attn

`python main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type normal --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --encoder_type gat_gt_parallel --fusion_type self_attn --run_name gat_gt_parallel-self_attn`

### gat_gt_parallel 和 gat_fusion

`python main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type normal --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --encoder_type gat_gt_parallel --fusion_type gat_fusion --run_name gat_gt_parallel-gat_fusion`

### gat_gt_parallel 和 gt_fusion

`python main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type normal --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --encoder_type gat_gt_parallel --fusion_type gt_fusion --run_name gat_gt_parallel-gt_fusion`

### 实验结果 - 编码器+图融合（交叉实验）

| Run | Encoder | Fusion | AUC | ACC | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| csglmd-basic | csglmd | basic | - | - | - | - | - |
| csglmd-dot | csglmd | dot | - | - | - | - | - |
| csglmd-additive | csglmd | additive | - | - | - | - | - |
| csglmd-self_attn | csglmd | self_attn | - | - | - | - | - |
| csglmd-gat_fusion | csglmd | gat_fusion | - | - | - | - | - |
| csglmd-gt_fusion | csglmd | gt_fusion | - | - | - | - | - |
| mgacmda-basic | mgacmda | basic | - | - | - | - | - |
| mgacmda-dot | mgacmda | dot | - | - | - | - | - |
| mgacmda-additive | mgacmda | additive | - | - | - | - | - |
| mgacmda-self_attn | mgacmda | self_attn | - | - | - | - | - |
| mgacmda-gat_fusion | mgacmda | gat_fusion | - | - | - | - | - |
| mgacmda-gt_fusion | mgacmda | gt_fusion | - | - | - | - | - |
| gat-basic | gat | basic | - | - | - | - | - |
| gat-dot | gat | dot | - | - | - | - | - |
| gat-additive | gat | additive | - | - | - | - | - |
| gat-self_attn | gat | self_attn | - | - | - | - | - |
| gat-gat_fusion | gat | gat_fusion | - | - | - | - | - |
| gat-gt_fusion | gat | gt_fusion | - | - | - | - | - |
| gt-basic | gt | basic | - | - | - | - | - |
| gt-dot | gt | dot | - | - | - | - | - |
| gt-additive | gt | additive | - | - | - | - | - |
| gt-self_attn | gt | self_attn | - | - | - | - | - |
| gt-gat_fusion | gt | gat_fusion | - | - | - | - | - |
| gt-gt_fusion | gt | gt_fusion | - | - | - | - | - |
| gat_gt_serial-basic | gat_gt_serial | basic | - | - | - | - | - |
| gat_gt_serial-dot | gat_gt_serial | dot | - | - | - | - | - |
| gat_gt_serial-additive | gat_gt_serial | additive | - | - | - | - | - |
| gat_gt_serial-self_attn | gat_gt_serial | self_attn | - | - | - | - | - |
| gat_gt_serial-gat_fusion | gat_gt_serial | gat_fusion | - | - | - | - | - |
| gat_gt_serial-gt_fusion | gat_gt_serial | gt_fusion | - | - | - | - | - |
| gat_gt_parallel-basic | gat_gt_parallel | basic | - | - | - | - | - |
| gat_gt_parallel-dot | gat_gt_parallel | dot | - | - | - | - | - |
| gat_gt_parallel-additive | gat_gt_parallel | additive | - | - | - | - | - |
| gat_gt_parallel-self_attn | gat_gt_parallel | self_attn | - | - | - | - | - |
| gat_gt_parallel-gat_fusion | gat_gt_parallel | gat_fusion | - | - | - | - | - |
| gat_gt_parallel-gt_fusion | gat_gt_parallel | gt_fusion | - | - | - | - | - |
