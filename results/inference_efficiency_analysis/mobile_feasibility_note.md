模型：经过R+U数据集微调的，Qwen3VL-2B模型，转换为MNN格式，未量化
设备：VIVO X300 pro，芯片联发科天玑9500，16G内存，1T储存
环境：模型端使用MNN Chat的APP加载，发布local host，使用cpu。客户端使用自己制作的app，调用摄像头，将图片与prompt一起发布到localhost，然后在听取到信息后记录延迟等信息同时再次发布下一帧图片与prompt。
实验结果：内存占用来自MNN Chat软件，延迟等来自我们自己制作的APP

Q&A：
Q1：为何选用Qwen3系列模型而非论文中普遍采用的Qwen2系列模型？
A1：我们尝试过使用Qwen2.5VL-3B和Qwen2VL-2B模型，其中Qwen2.5VL-3B模型在加载时经常会出现程序卡顿与闪退的情况，推测是占用内存太大。而Qwen2VL-2B系列模型在MNN Chat环境中存在输出错误等问题，导致模型无法使用，经排查发现为环境适配问而非模型本身问题。最终选用了较新的Qwen3VL-2B模型。
除此之外，官方提供的Qwen2.5VL-3B-Instruct-MNN模型（4bit量化）在MNN Chat中运行异常，无法用做仿真测试。

Q2：为何使用cpu而不是gpu或者npu？
A2：首先，对于npu，MNN Chat并不适配，且现在的开源开发工具普遍没有对手机的npu进行适配，考虑到开发会很花费时间，因此本次测试暂时放弃，我们将其作为Future Work。对于gpu，在MNN Chat中有一个选项是opencl，我们尝试使用这个方法进行测试，但是普遍token的输入输出速度要慢于cpu，我们之前使用高通骁龙8gen3芯片测试也是这种情况，再加上MNN Chat对天玑芯片的opencl适配并不好，会出现闪退情况，因此我们选取速度更快且更稳定的cpu进行测试。

Q3：占用内存太大的话，有没有尝试量化？
A3：有，但是经过我们尝试，我们自己微调过的多模态模型无法在转换成MNN格式时进行量化，主要是ViT无法通过官方工具量化的问题，两个手机都这样。最终只能选择bf16全精度转换。推测进行int4量化可以加快推理速度，我们未来将尝试解决这个问题。


---

Supplementary positioning note for paper writing:

- This folder records an exploratory mobile-side feasibility test, not the primary formal Inference Efficiency Analysis benchmark for the manuscript models.
- The mobile prototype uses a separately fine-tuned `Qwen3VL-2B` model because the manuscript-side `Qwen2.5-VL-3B` model was not stable in the current MNN Chat deployment environment and the available `Qwen2.5` mobile path was not usable for a clean end-to-end prototype run.
- Therefore, the latency and memory numbers in this folder may be cited only as supplementary engineering evidence that a phone-side prototype path is feasible under the present toolchain.
- These numbers must not be presented as direct on-device latency evidence for the manuscript's primary `Qwen2.5-VL-3B` or `Qwen2.5-VL-7B` models.
- In the reviewer response, the formal deployment evidence should still be the workstation-side latency/memory table for `Qwen2.5-VL-3B/7B`, while this mobile note can be used only to explain practical deployment constraints and future mobile optimization work.
