MNN Chat性能测评：
设备红米K Pad，芯片天玑9400+，16+1T
·Qwen3VL-2B-Instuct-fp16（Ours微调）
评测配置：后端-CPU，PP-128，TG-128
预填充速度：49.8 t/s ± 1.9
解码速度： 5.1 t/s ± 0.0
内存使用： 5GB
3 test，总时长：1.4 min

评测配置 OpenCL （闪退，推测是爆内存了）

·Qwen3VL-2B-Instruct-int4（官方模型，自己的模型int4会出问题）
评测配置：后端-CPU，PP-128，TG-128
预填充速度：93.6 t/s ± 13.9
解码速度： 20.1 t/s ± 1.9
内存使用： 1.8GB
3 test，总时长：23.374 s

评测配置：后端-OpenCL，PP-128，TG-128
预填充速度：260.6 t/s ± 70.6
解码速度： 23.0 t/s ± 0.7
内存使用： 3.4GB
3 test，总时长：18.272 s


任务app测评：
调用api→MNN Chat
Hazard：平均30.86秒

设备：VIVO X300 Pro，芯片天玑9500，16+1T
·Qwen3VL-2B-Instuct-fp16（Ours微调）
评测配置：后端-CPU，PP-128，TG-128
预填充速度：125.7 t/s ± 2.4
解码速度： 4.3 t/s ± 0.2
内存使用： 5GB
3 test，总时长：1.5 min


评测配置 OpenCL （闪退，推测是爆内存了）


任务app测评：
调用api→MNN Chat
Hazard：平均19.51秒

