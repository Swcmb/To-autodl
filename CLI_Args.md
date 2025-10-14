# EM 命令行参数详解

## 1. 入口与用法
- 入口脚本：`EM/main.py`
- 参数集中定义：`EM/parms_setting.py`（`settings()` 返回 `args`）
- 运行前置：`conda activate experiment`
- 运行示例：
  - 基本运行：`python EM/main.py`
  - 指定任务/数据/训练：
    ```
    python EM/main.py --task_type LDA \
      --in_file EM/dataset1/LDA.edgelist \
      --neg_sample EM/dataset1/non_LDA.edgelist \
      --epochs 50 --batch 64
    ```
  - 在线增强 + 多视图 MoCo：
    ```
    python EM/main.py --augment add_noise attribute_mask --augment_mode online --num_views 2
    ```
  - 控制并行：
    ```
    python EM/main.py --threads 16 --num_workers 8 --prefetch_factor 4
    ```
  - 命名运行：
    ```
    python EM/main.py --run_name my_try_001
    ```

## 2. 数据与任务配置
- `--in_file` [str]，默认 `dataset1/LDA.edgelist`  
  阳性样本（已知关联）文件路径；别名：`--file`
- `--neg_sample` [str]，默认 `dataset1/non_LDA.edgelist`  
  阴性样本（未知关联）文件路径
- `--task_type` `[LDA|MDA|LMI]`，默认 `LDA`  
  预测任务类型
- `--validation_type` `[5_cv1|5_cv2|5-cv1|5-cv2]`，默认 `5_cv1`  
  交叉验证类型；内部将 `5-cv1/5-cv2` 标准化为 `5_cv1/5_cv2`
- `--similarity_threshold` [float]，默认 `0.5`  
  构图的相似度阈值

## 3. 特征与增强
- `--feature_type` `[one_hot|uniform|normal|position]`，默认 `normal`  
  初始节点特征生成方式
- `--augment` [多选]，默认 `[random_permute_features]`  
  用于构造增强视图 `data_a`；可选：`none, random_permute_features, add_noise, attribute_mask, noise_then_mask`
- `--augment_mode` `[static|online]`，默认 `static`  
  `static`：每折固定增强；`online`：按 batch 动态增强（仅取 `augment` 列表第一个增强）
- `--noise_std` [float]，默认 `0.01`  
  `add_noise/noise_then_mask` 的高斯噪声标准差
- `--mask_rate` [float]，默认 `0.1`  
  `attribute_mask/noise_then_mask` 的列掩蔽比例
- `--augment_seed` `[int|None]`，默认 `None`  
  增强随机种子；`None` 时为 `seed+fold`；`online` 模式下每批派生 `seed+epoch*1000+iter`

## 4. 训练超参数
- `--seed` [int]，默认 `0`（NumPy/PyTorch/CUDA 同步）
- `--no-cuda` [flag]，默认 `False`（禁用 CUDA）
- `--lr` / `--learning_rate` [float]，默认 `5e-4`（`--learning_rate` 为 `--lr` 别名）
- `--dropout` [float]，默认 `0.1`
- `--weight_decay` [float]，默认 `5e-4`
- `--batch` [int]，默认 `25`
- `--epochs` [int]，默认 `1`

## 5. 多任务损失与别名
- `--loss_ratio1` [float]，默认 `1.0`（主任务 BCE）
- `--loss_ratio2` [float]，默认 `0.5`（对比 InfoNCE/CE）
- `--loss_ratio3` [float]，默认 `0.5`（第二对比，当前实现置 `0`）
- `--loss_ratio4` [float]，默认 `0.5`（节点级对抗 `BCEWithLogits`）
- 别名：
  - `--alpha` → `loss_ratio2`（默认 `0.5`）
  - `--beta` → `loss_ratio3`（默认 `0.5`）
  - `--gamma` → `loss_ratio4`（默认 `0.5`）

## 6. 模型与编码器
- `--dimensions` / `--embed_dim` [int]，默认 `256`（LMI 任务可用 `512`）
- `--hidden1` [int]，默认 `128`
- `--hidden2` [int]，默认 `64`
- `--decoder1` [int]，默认 `512`
- `--encoder_type` `[csglmd|csglmd_main|mgacmda|gat|gt|gat_gt_serial|gat_gt_parallel]`，默认 `csglmd`
- `--gat_heads` [int]，默认 `4`
- `--gt_heads` [int]，默认 `4`

## 7. 融合策略（pairwise interaction）
- `--fusion_type` `[basic|dot|additive|self_attn|gat_fusion|gt_fusion]`，默认 `basic`
- `--fusion_heads` [int]，默认 `4`（适用于 self-attn/GAT/GT 融合）

## 8. MoCo/对比学习
- `--moco_queue` [int]，默认 `4096`
- `--moco_momentum` [float]，默认 `0.999`
- `--moco_t` [float]，默认 `0.2`
- `--proj_dim` `[int|None]`，默认 `None`（默认随 `hidden2`）
- `--num_views` [int]，默认 `1`（单/多视图；训练/测试自动兼容）
- `--queue_warmup_steps` [int]，默认 `0`（前期仅 batch negatives）
- `--moco_debug` `[0|1]`，默认 `0`（轻量调试）

## 9. 并行与数据加载
- `--threads` [int]，默认 `32`  
  `main.py` 统一设置 OMP/MKL/OPENBLAS 等线程上限；Linux 可选 NUMA/亲和
- `--num_workers` [int]，默认 `-1`  
  DataLoader workers；`-1` 时自动 `min(8, threads)`，上限 `32`
- `--prefetch_factor` [int]，默认 `4`（仅 `num_workers>0` 生效）
- `--chunk_size` [int]，默认 `0`（自动 `20000`）
- `main.py` 自动导出环境变量：`EM_THREADS`, `EM_WORKERS`, `EM_CHUNK_SIZE`
- Linux 亲和/NUMA：默认启用 `EM_USE_NUMA=1`（若用户未设置）；也可设置 `EM_CPU_AFFINITY=1`
- CUDA 可见设备：代码固定 `os.environ["CUDA_VISIBLE_DEVICES"]="0"`

## 10. 数据保存与结果管理
- `--save_datasets` `[true|false]`，默认 `true`（是否保存构建数据集）
- `--save_format` `[npy|txt]`，默认 `npy`
- `--save_dir_prefix` [str]，默认 `result/data`（相对 `EM/`）
- 日志/结果目录：`result/data_时间戳`（支持 `run_name` 前缀）
- 5-fold 汇总与分折结果将保存为文本文件

## 11. 训练/验证中的注意点
- `--no-cuda` 仅影响使用与否，不改变 CUDA 可见性固定为 “0”
- `online` 增强仅使用 `augment` 的第一个策略；增强张量回到与 `data_a` 相同设备
- 多视图 MoCo：`train/test` 自动兼容单值/列表
- 对抗节点损失：按 `data_o.x.size(0)` 动态构造 `lbl2=[ones|zeros]` 并自动对齐设备

## 12. 常见组合示例
- LDA + 在线增强 + 双视图 MoCo：
  ```
  python EM/main.py --task_type LDA --augment add_noise attribute_mask --augment_mode online --num_views 2 --epochs 50 --batch 64
  ```
- GAT-GT 串联 + 自注意融合：
  ```
  python EM/main.py --encoder_type gat_gt_serial --fusion_type self_attn --fusion_heads 8
  ```
- 提高并行与预取：
  ```
  python EM/main.py --threads 24 --num_workers 8 --prefetch_factor 6
  ```

## 13. 其他
- Windows 下 `--shutdown` 无效（仅 Linux）
- 默认 `epochs=1`，正式训练请显式提升
- 推荐在 `experiment` Conda 环境中运行：`conda activate experiment`

# generate_experiment_report.py 使用说明

本脚本用于自动化生成实验参数笛卡尔积组合、按顺序执行实验（控制台实时输出）、解析 OUTPUT/result 目录下的结果文件，最终生成结构化 Markdown 报告与关键参数影响趋势图。

## 功能概览
- 参数网格：支持 6 类参数的笛卡尔积组合自动生成
- 顺序执行：可选择按组合顺序逐个运行，实时输出训练日志
- 结果解析：遍历 OUTPUT/result/*/result_summary_*.txt，鲁棒提取各类指标
- 报告生成：输出 EM/test.md，包含对比表、Top-5 最优推荐与趋势图（PNG）
- 可视化：图像保存于 OUTPUT/result/_report_assets，并在报告中引用
- 入口自动探测：优先使用 python EM/main.py，否则使用 python main.py

## 环境准备
- 建议在 Conda 虚拟环境 experiment 中运行：
  - Windows PowerShell: `conda activate experiment`

## 基础命令模板
脚本会自动探测入口文件：
- 若存在 `EM/main.py`，将 `python main.py …` 替换为 `python EM/main.py …`
- 否则保持 `python main.py …`

默认基础命令（可在脚本中修改 `BASE_STR_USER`）：
```
python main.py --file dataset1/LDA.edgelist --neg_sample dataset1/non_LDA.edgelist --validation_type 5-cv1 --task_type LDA --feature_type normal --similarity_threshold 0.5 --embed_dim 64 --learning_rate 0.0005 --weight_decay 0.0005 --epochs 3 --alpha 0.5 --beta 0.5 --gamma 0.5
```

## 参数说明
以下参数均可多值传入，脚本会做笛卡尔积组合：
- `--feature_types` 可选值：`one_hot uniform normal position`（默认 `normal`）
- `--augments` 可选值：`none random_permute_features add_noise attribute_mask noise_then_mask`（默认 `random_permute_features`）
- `--augment_modes` 可选值：`static online`（默认 `static`）
- `--encoder_types` 可选值：`csglmd csglmd_main mgacmda gat gt gat_gt_serial gat_gt_parallel`（默认 `csglmd`）
- `--fusion_types` 可选值：`basic dot additive self_attn gat_fusion gt_fusion`（默认 `basic`）
- `--queue_warmup_steps` 整数列表（默认 `0`）

控制参数：
- `--execute` 实际按组合顺序执行（不加则仅生成报告）
- `--epochs` 覆盖基础命令中的 epochs 值
- `--timeout` 单任务超时秒（`0` 不限时）
- `--limit` 只取前 N 个组合用于执行/清单（快速试跑）

## 运行示例
- 仅生成报告（扫描历史结果，不执行训练）：
```
python generate_experiment_report.py
```
- 小规模试跑并生成报告：
```
python generate_experiment_report.py --execute \
  --encoder_types csglmd mgacmda gat \
  --fusion_types basic dot additive \
  --augments random_permute_features none \
  --augment_modes static online \
  --feature_types normal one_hot \
  --queue_warmup_steps 0 10 \
  --epochs 3 --timeout 0
```
- 快速验证仅跑前 5 个组合：
```
python generate_experiment_report.py --execute --limit 5
```

## 输出物
- 报告：`EM/test.md`
  - 全部组合指标表（AUC、AUPRC、ACC、Precision、Recall、F1）
  - 最优配置推荐（按 AUC、ACC、F1 各取 Top-5）
  - 趋势图图片引用
- 可视化图片：`OUTPUT/result/_report_assets/*.png`

## 实验命名规则
每个组合生成唯一 `run_name`，便于和结果目录一致：
```
encoder-fusion__ft-{feature}__aug-{augment}__am-{mode}__qw-{steps}
```
示例：
```
csglmd-basic__ft-normal__aug-random_permute_features__am-static__qw-0
```

## 结果解析说明
脚本遍历 `OUTPUT/result/*/result_summary_*.txt` 并尝试解析如下键（大小写/别名兼容）：
- 识别字段：`run_name`, `encoder(_type)`, `fusion(_type)`, `feature_type`, `augment`, `augment_mode`, `queue_warmup_steps`
- 指标字段：`AUC/AUROC`, `AUPRC`, `ACC/Accuracy`, `Precision`, `Recall`, `F1/F1-Score`
若某些字段缺失，将尝试从目录名或 `run_name` 补全。

## 注意事项
- 执行前请确认 Python 与依赖已在 `experiment` 环境中
- 默认为“最后一个任务”自动追加 `--shutdown`
- 若 `matplotlib` 不可用，将跳过绘图但仍生成表格报告
- Windows 下默认使用 PowerShell 执行，控制台实时打印训练日志

## 常见问题
- 未生成图片：请安装 `matplotlib` 或使用有图形后端的环境（脚本已使用 `Agg` 保存 PNG）
- 未找到结果：确认 `OUTPUT/result` 下是否已有 `result_summary_*.txt`；或先加 `--execute` 运行一批
- 指标为 “-”：该字段在结果文件中未解析到，请检查 summary 文件的键名与格式