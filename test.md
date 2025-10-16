# 实验报告（控制变量法：moco × augment）

- 生成时间：2025-10-16 21:47:24
- 公共基础命令（入口已自动探测）：`python EM/main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type one_hot --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5`
- 固定组合数量：12
- 自变量：moco ∈ ['off', 'on']，augment ∈ ['random_permute_features']
- 常量：feature_type=one_hot，augment_mode=static，queue_warmup_steps=5

## 计划执行的命令（顺序）
- csglmd-gt_fusion__moco-off__aug-random_permute_features__ft-one_hot__am-static__qw-5
  - `python EM/main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type one_hot --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --augment_mode static --encoder_type csglmd --fusion_type gt_fusion --queue_warmup_steps 5 --augment random_permute_features --contrastive_type em --num_views 1 --run_name csglmd-gt_fusion__moco-off__aug-random_permute_features__ft-one_hot__am-static__qw-5`
- csglmd-gt_fusion__moco-on__aug-random_permute_features__ft-one_hot__am-static__qw-5
  - `python EM/main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type one_hot --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --augment_mode static --encoder_type csglmd --fusion_type gt_fusion --queue_warmup_steps 5 --augment random_permute_features --contrastive_type moco_multi --num_views 2 --run_name csglmd-gt_fusion__moco-on__aug-random_permute_features__ft-one_hot__am-static__qw-5`
- csglmd-basic__moco-off__aug-random_permute_features__ft-one_hot__am-static__qw-5
  - `python EM/main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type one_hot --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --augment_mode static --encoder_type csglmd --fusion_type basic --queue_warmup_steps 5 --augment random_permute_features --contrastive_type em --num_views 1 --run_name csglmd-basic__moco-off__aug-random_permute_features__ft-one_hot__am-static__qw-5`
- csglmd-basic__moco-on__aug-random_permute_features__ft-one_hot__am-static__qw-5
  - `python EM/main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type one_hot --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --augment_mode static --encoder_type csglmd --fusion_type basic --queue_warmup_steps 5 --augment random_permute_features --contrastive_type moco_multi --num_views 2 --run_name csglmd-basic__moco-on__aug-random_permute_features__ft-one_hot__am-static__qw-5`
- csglmd-self_attn__moco-off__aug-random_permute_features__ft-one_hot__am-static__qw-5
  - `python EM/main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type one_hot --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --augment_mode static --encoder_type csglmd --fusion_type self_attn --queue_warmup_steps 5 --augment random_permute_features --contrastive_type em --num_views 1 --run_name csglmd-self_attn__moco-off__aug-random_permute_features__ft-one_hot__am-static__qw-5`
- csglmd-self_attn__moco-on__aug-random_permute_features__ft-one_hot__am-static__qw-5
  - `python EM/main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type one_hot --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --augment_mode static --encoder_type csglmd --fusion_type self_attn --queue_warmup_steps 5 --augment random_permute_features --contrastive_type moco_multi --num_views 2 --run_name csglmd-self_attn__moco-on__aug-random_permute_features__ft-one_hot__am-static__qw-5`
- csglmd-additive__moco-off__aug-random_permute_features__ft-one_hot__am-static__qw-5
  - `python EM/main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type one_hot --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --augment_mode static --encoder_type csglmd --fusion_type additive --queue_warmup_steps 5 --augment random_permute_features --contrastive_type em --num_views 1 --run_name csglmd-additive__moco-off__aug-random_permute_features__ft-one_hot__am-static__qw-5`
- csglmd-additive__moco-on__aug-random_permute_features__ft-one_hot__am-static__qw-5
  - `python EM/main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type one_hot --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --augment_mode static --encoder_type csglmd --fusion_type additive --queue_warmup_steps 5 --augment random_permute_features --contrastive_type moco_multi --num_views 2 --run_name csglmd-additive__moco-on__aug-random_permute_features__ft-one_hot__am-static__qw-5`
- csglmd-gat_fusion__moco-off__aug-random_permute_features__ft-one_hot__am-static__qw-5
  - `python EM/main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type one_hot --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --augment_mode static --encoder_type csglmd --fusion_type gat_fusion --queue_warmup_steps 5 --augment random_permute_features --contrastive_type em --num_views 1 --run_name csglmd-gat_fusion__moco-off__aug-random_permute_features__ft-one_hot__am-static__qw-5`
- csglmd-gat_fusion__moco-on__aug-random_permute_features__ft-one_hot__am-static__qw-5
  - `python EM/main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type one_hot --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --augment_mode static --encoder_type csglmd --fusion_type gat_fusion --queue_warmup_steps 5 --augment random_permute_features --contrastive_type moco_multi --num_views 2 --run_name csglmd-gat_fusion__moco-on__aug-random_permute_features__ft-one_hot__am-static__qw-5`
- csglmd-dot__moco-off__aug-random_permute_features__ft-one_hot__am-static__qw-5
  - `python EM/main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type one_hot --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --augment_mode static --encoder_type csglmd --fusion_type dot --queue_warmup_steps 5 --augment random_permute_features --contrastive_type em --num_views 1 --run_name csglmd-dot__moco-off__aug-random_permute_features__ft-one_hot__am-static__qw-5`
- csglmd-dot__moco-on__aug-random_permute_features__ft-one_hot__am-static__qw-5
  - `python EM/main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type one_hot --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --augment_mode static --encoder_type csglmd --fusion_type dot --queue_warmup_steps 5 --augment random_permute_features --contrastive_type moco_multi --num_views 2 --run_name csglmd-dot__moco-on__aug-random_permute_features__ft-one_hot__am-static__qw-5`
- gt-self_attn__moco-off__aug-random_permute_features__ft-one_hot__am-static__qw-5
  - `python EM/main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type one_hot --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --augment_mode static --encoder_type gt --fusion_type self_attn --queue_warmup_steps 5 --augment random_permute_features --contrastive_type em --num_views 1 --run_name gt-self_attn__moco-off__aug-random_permute_features__ft-one_hot__am-static__qw-5`
- gt-self_attn__moco-on__aug-random_permute_features__ft-one_hot__am-static__qw-5
  - `python EM/main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type one_hot --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --augment_mode static --encoder_type gt --fusion_type self_attn --queue_warmup_steps 5 --augment random_permute_features --contrastive_type moco_multi --num_views 2 --run_name gt-self_attn__moco-on__aug-random_permute_features__ft-one_hot__am-static__qw-5`
- gt-gt_fusion__moco-off__aug-random_permute_features__ft-one_hot__am-static__qw-5
  - `python EM/main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type one_hot --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --augment_mode static --encoder_type gt --fusion_type gt_fusion --queue_warmup_steps 5 --augment random_permute_features --contrastive_type em --num_views 1 --run_name gt-gt_fusion__moco-off__aug-random_permute_features__ft-one_hot__am-static__qw-5`
- gt-gt_fusion__moco-on__aug-random_permute_features__ft-one_hot__am-static__qw-5
  - `python EM/main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type one_hot --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --augment_mode static --encoder_type gt --fusion_type gt_fusion --queue_warmup_steps 5 --augment random_permute_features --contrastive_type moco_multi --num_views 2 --run_name gt-gt_fusion__moco-on__aug-random_permute_features__ft-one_hot__am-static__qw-5`
- gat_gt_parallel-gt_fusion__moco-off__aug-random_permute_features__ft-one_hot__am-static__qw-5
  - `python EM/main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type one_hot --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --augment_mode static --encoder_type gat_gt_parallel --fusion_type gt_fusion --queue_warmup_steps 5 --augment random_permute_features --contrastive_type em --num_views 1 --run_name gat_gt_parallel-gt_fusion__moco-off__aug-random_permute_features__ft-one_hot__am-static__qw-5`
- gat_gt_parallel-gt_fusion__moco-on__aug-random_permute_features__ft-one_hot__am-static__qw-5
  - `python EM/main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type one_hot --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --augment_mode static --encoder_type gat_gt_parallel --fusion_type gt_fusion --queue_warmup_steps 5 --augment random_permute_features --contrastive_type moco_multi --num_views 2 --run_name gat_gt_parallel-gt_fusion__moco-on__aug-random_permute_features__ft-one_hot__am-static__qw-5`
- gat-self_attn__moco-off__aug-random_permute_features__ft-one_hot__am-static__qw-5
  - `python EM/main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type one_hot --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --augment_mode static --encoder_type gat --fusion_type self_attn --queue_warmup_steps 5 --augment random_permute_features --contrastive_type em --num_views 1 --run_name gat-self_attn__moco-off__aug-random_permute_features__ft-one_hot__am-static__qw-5`
- gat-self_attn__moco-on__aug-random_permute_features__ft-one_hot__am-static__qw-5
  - `python EM/main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type one_hot --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --augment_mode static --encoder_type gat --fusion_type self_attn --queue_warmup_steps 5 --augment random_permute_features --contrastive_type moco_multi --num_views 2 --run_name gat-self_attn__moco-on__aug-random_permute_features__ft-one_hot__am-static__qw-5`
- gat-gt_fusion__moco-off__aug-random_permute_features__ft-one_hot__am-static__qw-5
  - `python EM/main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type one_hot --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --augment_mode static --encoder_type gat --fusion_type gt_fusion --queue_warmup_steps 5 --augment random_permute_features --contrastive_type em --num_views 1 --run_name gat-gt_fusion__moco-off__aug-random_permute_features__ft-one_hot__am-static__qw-5`
- gat-gt_fusion__moco-on__aug-random_permute_features__ft-one_hot__am-static__qw-5
  - `python EM/main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type one_hot --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --augment_mode static --encoder_type gat --fusion_type gt_fusion --queue_warmup_steps 5 --augment random_permute_features --contrastive_type moco_multi --num_views 2 --run_name gat-gt_fusion__moco-on__aug-random_permute_features__ft-one_hot__am-static__qw-5`
- gat_gt_serial-gt_fusion__moco-off__aug-random_permute_features__ft-one_hot__am-static__qw-5
  - `python EM/main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type one_hot --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --augment_mode static --encoder_type gat_gt_serial --fusion_type gt_fusion --queue_warmup_steps 5 --augment random_permute_features --contrastive_type em --num_views 1 --run_name gat_gt_serial-gt_fusion__moco-off__aug-random_permute_features__ft-one_hot__am-static__qw-5`
- gat_gt_serial-gt_fusion__moco-on__aug-random_permute_features__ft-one_hot__am-static__qw-5
  - `python EM/main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type one_hot --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5 --augment_mode static --encoder_type gat_gt_serial --fusion_type gt_fusion --queue_warmup_steps 5 --augment random_permute_features --contrastive_type moco_multi --num_views 2 --run_name gat_gt_serial-gt_fusion__moco-on__aug-random_permute_features__ft-one_hot__am-static__qw-5 --shutdown`

### 对照表A：固定(组合+增强)，对比 moco(off/on)

| encoder-fusion | augment | moco=off(AUC/ACC/F1) | moco=on(AUC/ACC/F1) |
| --- | --- | --- | --- |
| csglmd-gt_fusion | random_permute_features | - | - |
| csglmd-basic | random_permute_features | - | - |
| csglmd-self_attn | random_permute_features | - | - |
| csglmd-additive | random_permute_features | - | - |
| csglmd-gat_fusion | random_permute_features | - | - |
| csglmd-dot | random_permute_features | - | - |
| gt-self_attn | random_permute_features | - | - |
| gt-gt_fusion | random_permute_features | - | - |
| gat_gt_parallel-gt_fusion | random_permute_features | - | - |
| gat-self_attn | random_permute_features | - | - |
| gat-gt_fusion | random_permute_features | - | - |
| gat_gt_serial-gt_fusion | random_permute_features | - | - |

### 对照表B：固定(组合+MoCo)，对比不同增强

| encoder-fusion | moco | augment | AUC | ACC | F1 |
| --- | --- | --- | --- | --- | --- |
| csglmd-gt_fusion | off | random_permute_features | - | - | - |
| csglmd-gt_fusion | on | random_permute_features | - | - | - |
| csglmd-basic | off | random_permute_features | - | - | - |
| csglmd-basic | on | random_permute_features | - | - | - |
| csglmd-self_attn | off | random_permute_features | - | - | - |
| csglmd-self_attn | on | random_permute_features | - | - | - |
| csglmd-additive | off | random_permute_features | - | - | - |
| csglmd-additive | on | random_permute_features | - | - | - |
| csglmd-gat_fusion | off | random_permute_features | - | - | - |
| csglmd-gat_fusion | on | random_permute_features | - | - | - |
| csglmd-dot | off | random_permute_features | - | - | - |
| csglmd-dot | on | random_permute_features | - | - | - |
| gt-self_attn | off | random_permute_features | - | - | - |
| gt-self_attn | on | random_permute_features | - | - | - |
| gt-gt_fusion | off | random_permute_features | - | - | - |
| gt-gt_fusion | on | random_permute_features | - | - | - |
| gat_gt_parallel-gt_fusion | off | random_permute_features | - | - | - |
| gat_gt_parallel-gt_fusion | on | random_permute_features | - | - | - |
| gat-self_attn | off | random_permute_features | - | - | - |
| gat-self_attn | on | random_permute_features | - | - | - |
| gat-gt_fusion | off | random_permute_features | - | - | - |
| gat-gt_fusion | on | random_permute_features | - | - | - |
| gat_gt_serial-gt_fusion | off | random_permute_features | - | - | - |
| gat_gt_serial-gt_fusion | on | random_permute_features | - | - | - |

### 明细：全部记录

| Run | encoder_type | fusion_type | moco | augment | feature_type | augment_mode | queue_warmup_steps | AUC | AUPRC | ACC | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

### 最优配置（按 AUC）

| Rank | AUC | Run | encoder_type | fusion_type | moco | augment |
| --- | --- | --- | --- | --- | --- | --- |

### 最优配置（按 ACC）

| Rank | ACC | Run | encoder_type | fusion_type | moco | augment |
| --- | --- | --- | --- | --- | --- | --- |

### 最优配置（按 F1）

| Rank | F1 | Run | encoder_type | fusion_type | moco | augment |
| --- | --- | --- | --- | --- | --- | --- |
