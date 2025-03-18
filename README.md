# AI_course_assessment

### Dataset: https://www.kaggle.com/datasets/gpreda/chinese-mnist/data
### Work: Compare the performance of CNN and Transformer

#### 超参数
1. **Epoch** = 10 / 20
2. **Learning Rate** = 0.0002
3. **Batch Size** = 64
4. **8Train Size : Test Size** = 4 : 1 

在单轮训练中，发现 CNN 需要设置较高的 **learning rate=0.001**，而 Transformer 需要设置较低的 **learning rate=0.0001**。为了保证对比实验的严谨性，这里设置两者均为 **0.0002**，效果不错.  
在 5折交叉验证中，会有一折出现未学习到图像特征进行随机猜测的情况，表现为 **loss=2.6-2.7, acc=6%-7%**  
