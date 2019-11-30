function Iout = warpping(Iin,x,y,M)
    %仿射变换函数 (旋转Rotation, 平移Translation, 调整Resize)
    %实际上就是在每一帧图像中找到最初的目标模块所在的区域

    % 计算变换后的坐标
    Tlocalx =  M(1,1) * x + M(1,2) *y + M(1,3);   %这里就是以角点为中心坐标把模板框T范围之内的x,y与图像的rows,cols联系起来
    Tlocaly =  M(2,1) * x + M(2,2) *y + M(2,3);
    %Iout  = interp2(Iin, Tlocalx, Tlocaly,'*linear');
    % 所有的邻域像素都涉及到线性插值.
    xBas0=floor(Tlocalx);      %该像素向前一位像素取整
    yBas0=floor(Tlocaly);
    xBas1=xBas0+1;             %该像素向后一位像素取整
    yBas1=yBas0+1;

    % 线性插值常数(百分比)
    xCom=Tlocalx-xBas0;
    yCom=Tlocaly-yBas0;
    perc0=(1-xCom).*(1-yCom);
    perc1=(1-xCom).*yCom;
    perc2=xCom.*(1-yCom);
    perc3=xCom.*yCom;

    % 将索引限制到边界
    check_xBas0=(xBas0<0)|(xBas0>(size(Iin,1)-1));    %先对所在的范围做判断
    check_yBas0=(yBas0<0)|(yBas0>(size(Iin,2)-1));
    xBas0(check_xBas0)=0;                 %如果超出图像的大小范围为真就置为0
    yBas0(check_yBas0)=0;
    check_xBas1=(xBas1<0)|(xBas1>(size(Iin,1)-1));
    check_yBas1=(yBas1<0)|(yBas1>(size(Iin,2)-1));
    xBas1(check_xBas1)=0;
    yBas1(check_yBas1)=0;

    Iout=zeros([size(x) size(Iin,3)]);
    for i=1:size(Iin,3);
        Iin_one=Iin(:,:,i);
        % Get the intensities
        intensity_xyz0=Iin_one(1+xBas0+yBas0*size(Iin,1));
        intensity_xyz1=Iin_one(1+xBas0+yBas1*size(Iin,1));
        intensity_xyz2=Iin_one(1+xBas1+yBas0*size(Iin,1));
        intensity_xyz3=Iin_one(1+xBas1+yBas1*size(Iin,1));
        Iout_one=intensity_xyz0.*perc0+intensity_xyz1.*perc1+intensity_xyz2.*perc2+intensity_xyz3.*perc3;
        Iout(:,:,i)=reshape(Iout_one, [size(x,1) size(x,2)]);
    end
end


