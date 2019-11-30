# Object-Tracking
Main use to store some object trackiing code 
KLT 是基于光流场的目标跟踪方法，在该代码中，主要是实现了对指定区域，也就是框选区域范围内进行角点检测，
并对角点响应最大的角点进行了跟踪，跟踪的显示结果就是对跟踪到的角点画'+',并以这个被跟踪到的角点的位置
为中心，50x50像素大小范围画矩形框。

主要包含了如下的文件：
       1、input：主要是一些连续用于跟踪的图像序列
       2、output：是我运行代码，并对指定区域进行框选后进行角点检测，对最大角点响应跟踪的结果展示
       3、KLT.m和warpping.m两个主要程序代码，运行时同时打开这两个代码文件，点击运行KLT.m主程序得到结果
       4、KLT.pdf是《good features to track》这篇论文，主要的代码实现原理和步骤就是出自于这篇论文
       
KLT主要代码实现步骤：
Setp1:  加载视频文件，或者视频图片序列
Setp2:  读入视频文件或者视频序列的第一帧图像，对图像中指定的区域检测角点  [pacth,rect] = imcrop(img)  rect()为想要跟踪的矩形区域范围内，保存角点的位置
Setp3:  以角点为中心，初始化定义warp范围T，包括初始化仿射变换参数p = [0 0 0 0 corners_i(1) corners_i(2)]
Setp4:  以初始的仿射变换参数，在第二帧图像中确定warp范围，Warp I with W_p得到I_warp
Setp5:  计算偏差，Subtract I from T，I_error= T - I_warped
Setp6:  针对得到的x，y方向的高斯卷积梯度图像后做仿射变换, Warp the gradient
Setp7:  计算雅可比矩阵，将雅可比矩阵与梯度相乘得到最快下降矩阵 I_steepest=gradient.*jacobian
Setp8:  计算Hessian 矩阵 H =  I_steepest(j2,:)‘ * I_steepest(j2,:)
Setp9:  误差与最快下降相乘，total = I_steepest(j3,:)‘*I_error(j3),计算delta_p=H/total
Setp10:  更新参数p，p = p + delta_p‘,显示在下一帧图像中被追踪到的角点的位置，我们也可以根据上一帧跟踪到的角点的位置每5帧图像对模板更行一次仿射变换参数p = [0 0 0 0 p(5) p(6)]
