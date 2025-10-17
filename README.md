# hyperparameter-tuning
模型超参数调优仓库

## 运行说明（三阶段）

- 阶段B：粗随机搜索（每任务 N=60，epochs=3）
  运行：
  ```
  python hyperparameter-tuning/hpo.py --stage B --tasks LDA MDA LMI --trials 60 --epochs 3
  ```
  输出目录：`experiments/hpo_YYYYMMDD_HHMMSS/{TASK}/`
  - opt_history_task_{TASK}.csv：所有 trial 的配置与 5-fold 指标
  - configs_top10_task_{TASK}.csv：按 AUPRC 主排序的 top10
  - best_configs_final.json：初版 top3 配置与复现命令
  - summary_task_{TASK}.md：阶段B摘要
  - error_report.txt：若有错误/OOM/Nan，写入摘要

- 阶段C：局部精调（epochs=20，启用 online 增强 与 mgraph 对抗）
  入口预置，后续将围绕阶段B top10 自动生成邻域搜索配置并执行：
  ```
  python hyperparameter-tuning/hpo.py --stage C --epochs_refine 20
  ```

- 最终复现：对每任务 top3 配置在 seeds=[0,1,2] 下复现
  入口预置：
  ```
  python hyperparameter-tuning/hpo.py --stage final --final_seeds 0 1 2
  ```

## 基线命令（示例）
与用户提供基线一致：
```
python model/main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type one_hot --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5
```

## 说明
- HPO 直接调用本目录的 load_data→Create_model→train_model 完成 5 折评估；不依赖外部库。
- 每个 trial 会同步覆盖 layer.args 中的关键字段，确保 EM 内部读取到当前试验的超参。
- 故障防护：NaN/Inf 或训练损失显著发散（首末epoch loss比>10×）将自动将学习率缩小 0.5 重试一次；OOM/未知异常写入 error_report.txt。
- Windows 环境下 DataLoader workers 固定为 0（autodl.py 已处理），整体按串行队列运行。
