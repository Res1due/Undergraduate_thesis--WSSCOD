# TrainPNet `TrainDDP.py` 文件说明

本文档说明 `code/TrainPNet/TrainDDP.py` 的核心功能、训练流程和关键参数。该脚本用于 **PNet 的分布式训练（DDP）**，并支持混合精度训练。

## 1. 文件作用

`TrainDDP.py` 负责完成以下任务：

- 初始化多卡分布式训练环境（NCCL + `mp.spawn`）。
- 构建 PNet 模型并封装为 `DistributedDataParallel`。
- 同时加载两类训练数据：
  - **fully**：有真实标注的数据；
  - **weakly**：由 ANet 生成伪标签的数据。
- 使用复合损失进行训练（主分割损失 + 边缘损失 + UAL 损失）。
- 每个 epoch 进行验证，按 MAE 保存最佳模型。

## 2. 主要模块结构

### 2.1 随机种子与混合精度

- `seed_torch(seed)`：固定 Python / NumPy / CUDA 的随机性，增强复现性。
- `scaler = amp.GradScaler(enabled=True)`：开启 AMP 梯度缩放。
- 前向计算在 `with amp.autocast(enabled=True):` 内执行，降低显存占用并提升吞吐。

### 2.2 损失函数

#### `NCLoss`

`NCLoss` 是 PNet 的主分割损失：

- `wbce_loss(...)`：边界加权 BCE（通过局部平均池化突出边界区域）。
- `forward(preds, targets, q)`：
  - 先做 sigmoid；
  - 使用 \|pred-target\|^q 构造鲁棒差异项；
  - 用预测与真值并集归一化；
  - 当 `q == 2` 时，返回 `loss.mean() + wbce`；
  - 当 `q == 1` 时，返回 `loss.mean() * 2`。

脚本中通过 `q_epoch` 控制训练前后阶段的 `q`：

- `epoch <= q_epoch`：`q=2`（更平滑）；
- `epoch > q_epoch`：`q=1`（更强调绝对差异）。

#### `dice_loss`

用于边缘分支监督（`preds[6:9]` 的加权组合）。

#### `cal_ual`（从工具函数导入）

不确定性感知损失，系数由 `get_coef(..., method='cos')` 动态调整：

- `ual_coef = get_coef(iter_percentage=i / total_step, method='cos')`
- `ual_loss = cal_ual(...) * ual_coef`

### 2.3 训练逻辑 `train(rank, world_size, opt)`

核心流程如下：

1. 初始化分布式进程组；
2. `Network()` -> `.to(device)` -> `DDP(...)`；
3. 构建两路训练集与各自 `DistributedSampler`：
   - `fully_train_loader`
   - `weak_train_loader`
4. 通过 `zip(fully_train_loader, weak_train_loader)` 同步取 batch，拼接为一次前向输入；
5. 计算损失：
   - `loss_init`：多尺度初级分割损失（`preds[0:4]`）
   - `loss_final`：最终分割损失（`preds[4]`）
   - `loss_edge`：边缘损失（`preds[6:9]`）
   - `ual_loss`：不确定性约束
6. 总损失：

```python
loss = loss_init + loss_final + loss_edge + 2 * ual_loss
```

7. 使用 `scaler.scale(loss).backward()` 与 `scaler.step(optimizer)` 更新参数；
8. rank0 负责日志、TensorBoard、保存模型和验证。

### 2.4 验证逻辑 `val(...)`

- 使用 `results[4]` 作为最终预测；
- 计算并记录：
  - `MAE`（主保存指标）；
  - `IoU`（辅助观测）；
- 当 `mae < opt.best_mae` 时保存 `Net_epoch_best.pth`。

## 3. 参数说明（`main()`）

重点参数如下：

- 学习率相关：
  - `--top_epoch`、`--top_lr`、`--init_lr`、`--min_lr`
- 训练规模：
  - `--epoch`、`--trainsize`、`--num_workers`
- 双数据流 batch：
  - `--batchsize_fully`（默认 6）
  - `--batchsize_weakly`（默认 24）
- 损失切换：
  - `--q_epoch`（默认 60）
- 数据路径：
  - `--fully_train_root`
  - `--weak_train_root`
  - `--val_root`
- 其他：
  - `--gpu_id`（如 `0,1,2,3`）
  - `--ration`（有标注数据比例，例如 `1` 表示 1% fully）
  - `--save_path`（权重和日志输出目录）

脚本会自动拼接路径：

- `fully_train_root = fully_train_root + ration + '%/'`
- `weak_train_root = weak_train_root + (100-ration) + '%/'`
- `save_path = save_path + ration + '%/'`

## 4. 分布式启动方式

入口使用：

```python
world_size = len(opt.gpu_id.split(','))
mp.spawn(train, args=(world_size, opt), nprocs=world_size, join=True)
```

并设置：

- `MASTER_ADDR=localhost`
- `MASTER_PORT=12355`

这意味着脚本会根据 `gpu_id` 的 GPU 数量，自动拉起对应训练进程。

## 5. 训练产物

在 `save_path` 目录下会生成：

- `log.log`：训练/验证日志；
- `summary/`：TensorBoard 数据；
- `Net_epoch_*.pth`：后期 epoch 权重（`epoch > 96`）；
- `Net_epoch_best.pth`：最佳 MAE 模型。

## 6. 注意事项

- `weak_train_root` 默认值中包含空格路径（`../pseduo label/...`），使用时需确认目录名是否一致。
- `min_lr` 默认写作 `14e-7`，等价于 `1.4e-6`，高于 `init_lr=1e-7`，如需严格衰减请按实验目标调整。
- 若多机/多任务同时训练，建议修改 `MASTER_PORT` 避免端口冲突。

## 7. 一句话总结

`code/TrainPNet/TrainDDP.py` 是一个面向噪声标注场景的 PNet 多卡训练脚本，通过 **fully + weakly 双数据流**、**NCLoss 动态阶段策略** 与 **AMP + DDP**，实现高效训练与稳定验证保存。
