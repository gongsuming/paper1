# paper1
说明：本次代码使用ResNet50作为Baseline，使用《TSM》作者提供的预训练参数
      在UCF101 RGB上的正确率： Baseline为89%；我们的DS为90%
      heat-map.py为注意力可视化代码
      为了验证注意力模块有效性，对2个视频进行可视化比较。
Baseline可视化结果：
![image](https://github.com/gongsuming/paper1/blob/main/Baseline-demo1.gif)
![image](https://github.com/gongsuming/paper1/blob/main/Baseline-demo2.gif)
我们的可视化结果：
![image](https://github.com/gongsuming/paper1/blob/main/Ours-demo1.gif)
![image](https://github.com/gongsuming/paper1/blob/main/Ours-demo2.gif)
从gif结果来看，第一个视频两者表现差不多，我们的DS略好；在第二个视频上，我们的DS明显优于Baseline。
由于GitHub的原因，gif图显示不了，有以下2个解决方法：
1> 将本项目下载到本地。
2> 连上VPN，再看此项目。


本文DS在UCF101上的训练结果PTH文件：（其余自行训练即可）
链接：https://pan.baidu.com/s/1mXSKThCZOc-QHgp3tbAtuQ 
提取码：cork 

代码已公布，数据路径这些设置需要按自己情况配置。
详细介绍可参考https://github.com/mit-han-lab/temporal-shift-module。
感谢《TSM》作者开源项目的帮助。
