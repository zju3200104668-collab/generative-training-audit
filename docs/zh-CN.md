# 生成模型训练静默故障审计

这个仓库将四类生成模型训练故障整理成最小复现和自动测试：

| 故障 | 表面现象 | 真正需要验证的对象 |
|---|---|---|
| Scheduler 轨迹错位 | 配置显示两步 | 实际执行的时间与 sigma 转移 |
| BF16 EMA 冻结 | EMA 每步都调用 | 更新量是否超过当前数值附近的表示间隔 |
| Teacher cache 噪声错配 | target 能正常加载 | target 与当前输入是否对应同一噪声张量 |
| GAN 梯度断路 | fake loss 能正常打印 | fake-only loss 是否对判别器产生非零梯度 |

## 运行

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"

python examples/ema_freeze_demo.py
python examples/noise_pairing_demo.py
python examples/scheduler_trace_demo.py
python examples/gan_detach_demo.py
pytest -q
```

## 1. Scheduler：记录真实转移，而不是相信配置名

`num_inference_steps=2` 只描述接口参数，不能单独证明模型执行了预期的两次跳转。审计时应记录：

```text
(t_i, sigma_i) -> (t_{i+1}, sigma_{i+1})
```

并检查起点、终点、转移数和轨迹连续性。`scheduler.py` 不绑定某一个具体调度器，而是提供记录真实轨迹后可复用的 invariant 检查。

## 2. EMA：慢变量应保留 FP32 master

EMA 更新为：

$$
\bar\theta \leftarrow \bar\theta + (1-\beta)(\theta-\bar\theta).
$$

当右侧增量小于 BF16 在 $\bar\theta$ 附近的表示间隔时，低精度写回可能再次舍入到旧值。安全做法是让 shadow 参数始终保留 FP32，仅在模型使用或通信时转换精度。

## 3. Noise pairing：随机变量也是数据契约

若 teacher target 为 $G_T(c,\epsilon_a)$，student 输入却使用 $\epsilon_b$，逐样本回归实际上变成：

$$
d(G_S(c,\epsilon_b),G_T(c,\epsilon_a)),\qquad \epsilon_a\ne\epsilon_b.
$$

因此 cache 中不仅应记录文本、target 和 sample id，还应记录确定性 seed 或噪声指纹。`noise.py` 使用 shape、dtype 和连续字节的 SHA-256 指纹做启动期断言。

## 4. GAN：detach sample，不要 detach fake logits

更新判别器时，应阻止梯度进入生成器，同时保留 fake loss 到判别器的路径：

```text
正确：fake.detach() -> D -> fake loss
错误：fake -> D -> fake_logits.detach() -> fake loss
```

错误写法仍然能算出有限的 fake loss，但其对判别器参数的梯度严格为零。仓库通过 fake-only backward 直接检查这一点。

## 使用边界

这些代码用于说明审计方法，不复现任何具体企业训练系统，也不包含模型权重、内部代码、数据或配置。生产环境仍应针对所用框架、硬件和 scheduler 增加端到端断言。
