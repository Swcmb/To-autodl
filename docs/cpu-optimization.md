# CPU 并行优化与性能指南

本项目已针对多核 CPU 场景完成系统级优化，适用于 16–32 逻辑核、120GB 内存级别的机器（更大机器亦可运行，但默认做了并发上限与亲和控制，避免过度超订阅与跨 NUMA 抖动）。

## 一、默认参数与行为

- 线程与并行
  - EM_THREADS=32（自动探测并上限32）
  - EM_WORKERS=8（用于数据加载与部分CPU任务）
  - EM_CHUNK_SIZE=20000（CPU计算任务的缺省切块大小）
- Linux NUMA/亲和（默认开启）
  - EM_USE_NUMA=1（仅 Linux 生效）：主进程绑定到 node0 或前 EM_THREADS 个逻辑核，减少跨 NUMA 迁移
- 训练保持在 GPU 上进行，不调整 torch CPU 线程数，避免干扰 GPU 吞吐
- 相似度阶段优化
  - RBF（高斯核）全向量化，使用 BLAS/OMP 多线程
  - PBPA 功能相似度：多进程并行 + 预计算索引 + 进程 initializer 注入只读缓存 + 仅传 (i,j) 降低序列化开销
- DataLoader
  - num_workers=8、prefetch_factor=4、persistent_workers（当 workers>0）

## 二、关键命令行参数

- --threads：底层数值库线程上限，-1 表示自动探测（默认32封顶）
- --num_workers：DataLoader/CPU任务辅助进程数，-1 自动（默认 min(8, threads)）
- --prefetch_factor：DataLoader 预取因子（仅 workers>0 有效）
- --chunk_size：CPU计算的块大小（默认 20000）
- --use_numa：CLI 级开关（可选）；同时也支持环境变量 EM_USE_NUMA=1（已默认开启）

建议默认组合：
- --threads 32 --num_workers 8 --prefetch_factor 4 --chunk_size 20000
- PBPA 进程数自适配：eff_workers=min(8, threads//4)（代码中已实现）

## 三、快速复现实验

- 快速校验（1 epoch）：
  - python main.py --run_name final-check
- 性能参数明确指定（覆盖默认）：
  - python main.py --threads 32 --num_workers 8 --prefetch_factor 4 --chunk_size 20000 --epochs 1 --run_name cpu32
- 采集分段计时（日志含 [TIMING] 段落）：
  - python main.py --threads -1 --num_workers -1 --prefetch_factor 4 --epochs 1 --run_name profiling

说明：
- [TIMING] Fold X similarity stage: S.s 表示相似度阶段耗时（项目中主要瓶颈已优化）
- graph/features/DataLoader 的耗时通常较小

## 四、Linux NUMA/CPU 亲和

- 默认开启：EM_USE_NUMA=1（main.py 已设默认，仅 Linux 生效）
- 行为：尝试绑定到 node0 下的逻辑核；如果未检测到 NUMA 信息，则回退绑定到前 threads 个 CPU
- 关闭方式：运行前将 EM_USE_NUMA 和 EM_CPU_AFFINITY 置空或 0

示例：
- 仅使用环境变量（推荐）：
  - EM_USE_NUMA=1 python main.py --threads 32 ...
- 使用 CLI（可选）：
  - python main.py --use_numa --threads 32 ...

注意：
- 若系统未安装 psutil，将回退使用 os.sched_setaffinity（若内核支持）

## 五、CPU 压力测试脚本（生成 CSV）

脚本：cpu_parallel_stress_test.py
- 用途：遍历 threads × workers × chunk_size 参数矩阵，模拟 RBF + PBPA 风格负载，输出 CSV 到 OUTPUT/result
- 示例：
  - python cpu_parallel_stress_test.py --threads_list 8,16,32 --workers_list 2,4,8 --chunk_list 10000,20000
- 脚本输出：
  - 控制台：[STRESS] threads=... workers=... chunk=... rbf=...s pbpa=...s total=...s
  - CSV：OUTPUT/result/cpu_stress_YYYYMMDD_HHMMSS.csv

将 CSV 纳入报告/README：
- 在报告章节“CPU 参数基准”中插入表格/图表，可由 CSV 生成（例如用 pandas + matplotlib 或表格粘贴）
- 推荐列：threads, workers, chunk_size, rbf_s, pbpa_s, total_s
- 选择 total_s 最小的一组作为推荐值（通常 32/8/20000 在多数场景表现良好）

## 六、常见问题与排查

- Pickling 错误：Can't pickle local object
  - 原因：将 worker 函数定义在函数内部；已修复为模块级函数
- Windows 平台多进程
  - 需在 if __name__ == "__main__": 防护下启动；本项目主要在 Linux 环境验证
- psutil 不存在
  - 仅影响亲和设置；会自动回退 os.sched_setaffinity 或忽略
- 线程风暴
  - 已统一设置 OMP/MKL/OPENBLAS/NUMEXPR/BLIS/VECLIB_MAXIMUM_THREADS；如仍出现抖动，先下调 --threads 或 --num_workers

## 七、推荐使用流程

1. 按默认参数运行一次 5 折训练，确认指标与耗时
2. 若机器规格不同，执行压力测试脚本，找到本机最佳 threads/workers/chunk
3. 将最佳参数固化到命令行或环境变量（EM_THREADS/EM_WORKERS/EM_CHUNK_SIZE）
4. 如需进一步优化，请优先关注 [TIMING] 中 similarity stage 段落（其余阶段收益有限）

---

## 八、你的机器上的推荐参数组合

- 选取标准：按 total_s 最小
- 最优组合：
  - threads=32, workers=8, chunk_size=10000
  - rbf_s=0.0556s, pbpa_s=0.1048s, total_s=0.1605s
- 说明：
  - 在你的机器上，chunk_size=10000 略优于 20000（更小的任务块降低单批调度等待）
  - 线程与 workers 的组合 32 × 8 性能最佳，符合我们对 eff_workers=min(8, threads//4) 的经验法则
- 推荐命令：
  - python main.py --threads 32 --num_workers 8 --prefetch_factor 4 --chunk_size 10000 --run_name cpu32-best
