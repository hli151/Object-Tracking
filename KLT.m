clc;
clear all;
close all;

%% %%%%%% Begin Load Images or Video Frames%%%%%%%%%%%%%
%%对应加载的是图像序列
%读入图像，首先找到图像所在的文件夹名字和图像的格式
imPath = 'input'; imExt = 'jpg';

%%%%% Load the images
%=======================
% 检查文件目录是否存在
if isdir(imPath) == 0
    error('User error : The image directory does not exist');
end

filearray = dir([imPath filesep '*.' imExt]); % 获取文件目录下的所有图片
NumImages = size(filearray,1); % 图片的数量
if NumImages < 0
    error('No image in the directory');
end

disp('Loading image files from the video sequence, please be patient...');
% Get image parameters
imgname = [imPath filesep filearray(1).name]; % 获得序列中的图像的name
I = imread(imgname);
if size(I)==3
   I = rgb2gray(I);
end
VIDEO_WIDTH = size(I,2);
VIDEO_HEIGHT = size(I,1);

ImSeq = zeros(VIDEO_HEIGHT, VIDEO_WIDTH, NumImages);
for i=1:NumImages
    imgname = [imPath filesep filearray(i).name]; % 获得序列中的图像的name
    img = imread(imgname);
    if size(img)==3
        img = rgb2gray(img);
    end
    ImSeq(:,:,i) = img; % 加载进图像
end
disp(' ... OK!');
%%对应加载的是视频
%{
Video_name = 'name.type'; %name.type对应你要加载的视频文件的名字
vid = VideoReader(Video_name);
NumImages = vid.NumberOfFrames(vid);  %视频文件所包含的全部帧数
Height = vid.Height;    %对应图像帧的Height
Width = vid.Width;      %对应图像帧的Width
ImSeq = zeros(Height, Width, NumImages);
disp('Loading image files from the video sequence, please be patient...');
for i = 1:NumImages
    img = read(vid,1);
    if size(img) == 3
        img = rgb2gray(img);
    end
    ImSeq(:,:,i) = img;
end
I = ImSeq(:,:,1);
disp(' ... OK!');
%}
%% %%%%%%%%%%%End The Load %%%%%%%%%%%%%%%%%%
%% 框选最初的跟踪目标
%You can manual initialization use the function imcrop
[pacth,rect] = imcrop(ImSeq(:,:,1)./255);    %框选出来的矩形框区域的最左上角的位置为（rect(2),rect(1)）;宽为rect(3),高为rect(4);
%ROI_Center = round([rect(1)+rect(3)/2 , rect(2)+rect(4)/2]); 
%ROI_Width = rect(3);
%ROI_Height = rect(4);
%% Harris角点的检测
%1、梯度计算
%2、矩阵形成
%3、特征值计算
%通过计算x和y方向的平滑（使用高斯函数）梯度来检测给定灰度图像的角点
%%%%%%%%%%% Start Harris Corners  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
threshold=0.36;
I = double(I);
sigma=2; 
k = 0.04;             %在Harris角点检测中通常k = 0.04
dx = [-1 0 1; -1 0 1; -1 0 1]/6;%导数模板
dy = dx';                       %dy等于dx的转置
Ix = conv2(I, dx, 'same');      %卷积计算x方向一次梯度
Iy = conv2(I, dy, 'same');      %卷积计算y方向一次梯度
g = fspecial('gaussian',fix(6*sigma), sigma); %Gaussian 滤波，fix()函数表示向靠近零取整
Ix2 = conv2(Ix.^2, g, 'same');  %卷积计算x方向二次梯度
Iy2 = conv2(Iy.^2, g, 'same');  %卷积计算y方向二次梯度
Ixy = conv2(Ix.*Iy, g,'same');  
R= (Ix2.*Iy2 - Ixy.^2) - k*(Ix2 + Iy2).^2;   %得到矩阵的响应函数,选取局部最大的响应的R为角点
%使得R归一化 ，0到1之间 
minr = min(min(R));
maxr = max(max(R));
R = (R - minr) / (maxr - minr);

%%只对被框选的矩形区域的角点检测
Rm = zeros(size(R,1),size(R,2));
Rm(rect(2):(rect(2)+rect(4)),rect(1):(rect(1)+rect(3))) = R(rect(2):(rect(2)+rect(4)),rect(1):(rect(1)+rect(3)));%框选出来的矩形框区域的最左上角的位置为（rect(2),rect(1)）;宽为rect(3),高为rect(4);
%计算阈值5×5窗口上R的局部最大值
maxima = ordfilt2(Rm, 25, ones(5));   %二维统计顺序滤波函数ordfilt2函数，对模板中的对应像素非零像素值做一个从小到大的顺序排列，这里相当于提取5×5窗口上R的局部最大值
mask = (Rm == maxima) & (Rm > threshold);
maxima = mask.*R;

figure(1);
colormap('gray');       %用map矩阵映射当前图形的色图。
imagesc(I);
hold on;
[r,c] = find(maxima>0);  %find（A）返回矩阵A中非零元素所在位置，找到角点的位置并赋给了[r,c]，但是对应的行和列交换了;
plot(c,r,'*');           %所以后面在画出跟踪到的角点的位置的时候是相当于(c,r),也就是后面对应的(p(6),p(5)).
hold off;
%saveas(gcf,'mainCornersSeq1.jpg');
%保存角点
[L ~ ]=size(c);
corners = cell(1,L);    %定义一个数组区域来存储角点，具体是角点的数量
%选择检测出的角点的最大响应值的角点，并保存该角点的位置
temp = I(c(1),r(1));
tempL = [r(1),c(1)];   %这里是为了和下面保持一致，而没有直接写成（c(1),r(1))的形式
for i=2:L
    if (I(c(i),r(i)) > temp)           %这里具体只是保留一个角点的位置，也就是方便我们观察的角点的位置
        temp = I(c(i),r(i));
        tempL = [r(i),c(i)];
    end
end
corners_i = tempL;
%% %%%%%%%%%  End Harris Corners  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% %%%%%%%%% Start KLT Tracker %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%windowSize = 30;
windowSize = 20;
[rows, cols, chan] = size(I);

if (corners_i(1)-windowSize > 0 && corners_i(1)+windowSize <= rows && corners_i(2)-windowSize > 0 && corners_i(2)+windowSize < cols)
        %定义初始的仿射变换参数，就是直接以角点为中心参考，没有任何的平移，旋转，缩放
        p = [0 0 0 0 corners_i(1) corners_i(2)];
        cornerCounter = 0;
        newCornerCounter = 1;
        T = I(corners_i(1)-windowSize:corners_i(1)+windowSize,corners_i(2)-windowSize:corners_i(2)+windowSize);  %T表示以角点为中心，2*windowsize大小范围内的矩形框
        T= double(T);
        %Make all x,y indices
        [x,y]=ndgrid(0:size(T,1)-1,0:size(T,2)-1);  %ndgrid函数实现多维数组的全排列，即是使得在T的大小范围内所有得位置都可能随机搭配组合
         %计算模板图像的中心
        TemplateCenter=size(T)/2;
        %使模板图像的中心坐标为（0,0）
        x=x-TemplateCenter(1); y=y-TemplateCenter(2);    %用T模板中得像素位置（x,y）-(TemplateCenter(1),TemplateCenter(2)),使得模板中心的坐标为(0,0)
     for n = 2:NumImages
            NextFrame = ImSeq(:,:,n);
            NextFrameCopy = NextFrame;
            if(size(NextFrame) == 3)    %如果是三通道的RGB图像的话，就转化为rgb2gray的灰度图像
                NextFrame = rgb2gray(NextFrame);
            end
            copy_p = p;
            I_nextFrame= double(NextFrame); 
            delta_p = 7;
            sigma = 3;
            %Make derivatives kernels
            [xder,yder]=ndgrid(floor(-3*sigma):ceil(3*sigma),floor(-3*sigma):ceil(3*sigma));
            DGaussx=-(xder./(2*pi*sigma^4)).*exp(-(xder.^2+yder.^2)/(2*sigma^2));
            DGaussy=-(yder./(2*pi*sigma^4)).*exp(-(xder.^2+yder.^2)/(2*sigma^2));
            % 卷积得到一阶导数
            Ix_grad = imfilter(I_nextFrame,DGaussx,'conv');     %得到X方向的高斯卷积梯度
            Iy_grad = imfilter(I_nextFrame,DGaussy,'conv');     %得到Y方向的高斯卷积梯度
            counter = 0;
            %设定阈值为0.01，小于阈值视为条件不满足，跳出while循环，视为已经找到跟踪的目标，大于阈值视为条件满足，执行while循环
            Threshold = 0.01;
            while ( norm(delta_p) > Threshold) %norm(A)表示返回矩阵A中最大的奇异值
                counter= counter + 1;
                %如果超过80个循环不收敛，则中断，并将其视为收敛
                if(counter > 80)
                    break;
                end
                %norm(delta_p)
                %模板旋转和平移的仿射矩阵
                W_p = [ 1+p(1) p(3) p(5); p(2) 1+p(4) p(6)];
                %1 Warp I with W_p
                I_warped = warpping(I_nextFrame,x,y,W_p);      %
                %2 用之前建立的目标模板减去在下一帧中warp找到的区域范围，Subtract I from T
                I_error= T - I_warped;
                % Break if outside image
                if((p(5)>(size(I_nextFrame,1))-1)||(p(6)>(size(I_nextFrame,2)-1))||(p(5)<0)||(p(6)<0)), break; end; %这里也要对划定的模板范围做判断，是否在下一帧的图像中
                %3 Warp the gradient
                Ix =  warpping(Ix_grad,x,y,W_p);   %针对得到的x方向的高斯卷积梯度图像后做仿射变换
                Iy = warpping(Iy_grad,x,y,W_p);    %针对得到的y方向的高斯卷积梯度图像后做仿射变换
                %4 计算雅可比矩阵
                W_Jacobian_x=[x(:) zeros(size(x(:))) y(:) zeros(size(x(:))) ones(size(x(:))) zeros(size(x(:)))];
                W_Jacobian_y=[zeros(size(x(:))) x(:) zeros(size(x(:))) y(:) zeros(size(x(:))) ones(size(x(:)))];
                %5 计算最快下降
                I_steepest=zeros(numel(x),6);
                for j1=1:numel(x),
                    W_Jacobian=[W_Jacobian_x(j1,:); W_Jacobian_y(j1,:)];
                    Gradient=[Ix(j1) Iy(j1)];
                    I_steepest(j1,1:6)=Gradient*W_Jacobian;    %
                end
                %6 计算 Hessian 矩阵
                H=zeros(6,6);
                for j2=1:numel(x), H = H + I_steepest(j2,:)'*I_steepest(j2,:); end
                %7 误差与最快下降相乘
                total=zeros(6,1);
                for j3=1:numel(x), total = total + I_steepest(j3,:)'*I_error(j3); end
                %8 计算 delta_p
                delta_p=H\total;
                %9 更新参数 p 
                 p = p + delta_p';  
            end
            cornerCounter = cornerCounter+1;
            %在大括号后面加上一逗号，表示加入这一段代码，每5帧图像更新一次模板，以上一帧图像中跟踪到的图像中的角点的位置为新的模板图像的中心
            %如果把大括号后面的逗号去掉，则表示屏蔽掉这段代码，就是只以最初的目标模板进行跟踪，不对模板模板更新
            %{，
            if (cornerCounter == 5)     %每5帧更新一次模板T
                T = NextFrameCopy(p(5)-windowSize:p(5)+windowSize,p(6)-windowSize:p(6)+windowSize);
                p = [0 0 0 0 p(5) p(6)];
                T= double(T);
                %Make all x,y indices
                [x,y]=ndgrid(0:size(T,1)-1,0:size(T,2)-1);
                %计算模板图像的中心
                TemplateCenter=size(T)/2;
                %使得模板图像的中心的坐标为(0,0)
                x=x-TemplateCenter(1); y=y-TemplateCenter(2);
                cornerCounter = 0;
            end
            
            %newCornerCounter = newCornerCounter+1;
            %}
    disp('被跟踪到的角点的位置为:\n');
    fprintf('%d,%d',p(6),p(5));
    figure(2),subplot(1,1,1), imshow(NextFrame, []);         %逐帧显示下一帧图像
    hold on;
	plot(p(6), p(5), '+', 'Color', 'r', 'MarkerSize',10);     %画出追踪到的角点的位置（p(6),p(5)）,'+'
    rectangle('Position',[p(6)-25 p(5)-25 50 50],'LineWidth',2,'EdgeColor','r');   %以追踪到的角点（p(6),p(5)）为中心画矩形框
    drawnow;
    end 
end

%% %%%%%%%%% End KLT Tracker %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
 

