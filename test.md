## 测试与运行说明

本文件用于补充命令行示例，便于快速复现实验。

### 增强使用示例

- 每折静态增强（默认）：random_permute_features
```
conda activate experiment
python main.py --in_file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist \
  --validation_type 5_cv1 --task_type LDA --feature_type normal \
  --augment random_permute_features --augment_mode static --seed 0 --run_name em_static_perm
```

- 每批在线增强：以组合增强为例（noise_then_mask）
```
conda activate experiment
python main.py --in_file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist \
  --validation_type 5_cv1 --task_type LDA --feature_type normal \
  --augment noise_then_mask --noise_std 0.01 --mask_rate 0.1 \
  --augment_mode online --augment_seed 0 --seed 0 --run_name em_online_combo
```

参数说明：
- --augment：none | random_permute_features | add_noise | attribute_mask | noise_then_mask
- --augment_mode：static（每折静态增强，仅构造 data_a 时应用）| online（训练阶段每 batch 动态增强，仅作用 data_a）
- --noise_std：add_noise/noise_then_mask 的噪声强度（默认 0.01）
- --mask_rate：attribute_mask/noise_then_mask 的列掩码比例（默认 0.1）
- --augment_seed：增强随机种子；static 模式为空时将派生为 seed+fold；online 模式用于生成每 batch 派生种子