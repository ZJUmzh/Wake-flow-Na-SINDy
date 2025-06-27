%% 遗传算法主程序
% 这是1.1版本，主要增加了拉丁超立方抽样
clear
clc
close all

npop=100;%种群个数

nvar=2;%参数量
maxit=100;%迭代次数

pc1=0.8;%交叉率
pc2=0.6;
pm1=0.2;%变异率
pm2=0.1;

r = 2;
load PODcoefficients
x = [alpha(1:5001,1:r) alphaS(1:5001,1)];
load PODcoefficients_run1
x1 = [alpha(1:3000,1:r) alphaS(1:3000,1)];

rng(0);

template.x=[];
template.y=[];
template.dominationset=[];%支配集，指个体能够支配的其他个体所对应的下标
template.dominated=[];%被支配数，指个体在自然选择中被支配的次数
template.rank=[];%等级，指个体在自然选择中所占据的生态位
template.cd=[];%拥挤度
 
pop=repmat(template,npop,1);
%% 二阶回归拟合
%分别对lambda和k进行优化
lb = [1,-8];%下界
ub = [4,0];%上界
%% 交叉变异迭代
lhs_rand = lhsdesign(npop,2);
for i=1:npop
    %随机生成参数组合
    pop(i).x = lb+lhs_rand(i,:).*(ub-lb);
    pop(i).y=respond(pop(i).x(1),pop(i).x(2),x,x1);%根据参数组合得到对应的y值
end

mean_iteration = [];%迭代过程记录
iteration = [];
n_best = 0;%记录最优值是同一个的次数
y_best = zeros(1,maxit+1);%记录最优解
%for it=1:maxit
it = 1;
while n_best <= maxit/4 & length(iteration) <= maxit
    tic
    npc=1;
    popc=repmat(template,npop/2,2);
    fall = [pop.y];%fall存储所有的y值
    fall = reshape(fall,[2, npop]);%第一行存储所有的y1，第二行存储所有的y2
    fmax =2; %这一代中综合y值最高的个体，以其作为归一化的上限
    fave =mean(fall(1,:))./max(fall(1,:))+mean(fall(2,:))./max(fall(2,:)); %这一代中综合y值的平均数
    %% 交叉
    for i=1:npop/2
        ind=randperm(npop,2);%选择交叉个体

        fcross1=pop(ind(1)).y(1)./max(fall(1,:))+pop(ind(1)).y(2)./max(fall(2,:));
        fcross2=pop(ind(2)).y(1)./max(fall(1,:))+pop(ind(2)).y(2)./max(fall(2,:));
        fcross = max([fcross1 fcross2]);
        pc=pc2;
        if(fcross>fave)
            pc = pc1-(pc1-pc2).*(fcross-fave)./(fmax-fave);
        end

        value = rand();
        if(value<=pc)
        [popc(npc,1).x,popc(npc,2).x]=Cross(pop(ind(1)).x,pop(ind(2)).x);%交叉得到新的参数组合
        popc(npc,1).y=respond(popc(npc,1).x(1),popc(npc,2).x(2),x,x1);  
        popc(npc,2).y=respond(popc(npc,2).x(1),popc(npc,2).x(2),x,x1);
        npc=npc+1;
        end
    end

    npc = npc-1;
    popc(npc+1:npop/2,:)=[];

    %% 变异
    npm=1;
    popm=repmat(template,npop,1);
    i_mutate = [];%存储变异个体的编号
    for j=1:npop%
        ind=randperm(npop,1);%选择变异个体
        fmutate=pop(ind(1)).y(1)./max(fall(1,:))+pop(ind(1)).y(2)./max(fall(2,:));
        pm=pm2;
        if(fmutate>fave)
            pm = pm1-(pm1-pm2).*(fmutate-fave)./(fmax-fave);
        end
        value=rand();
        if(value<=pm)
            i_mutate = [i_mutate ind];%存储变异个体
        end
    end
    mutate_rand = lhsdesign(length(i_mutate),1);%根据变异个体数量进行拉丁超立方抽样
    for j = i_mutate
        popm(npm,1).x=Mutate(pop(j).x,lb,ub,mutate_rand(npm));
        popm(npm,1).y=respond(popm(npm).x(1),popm(npm).x(2),x,x1);
        npm=npm+1;
    end

    npm=npm-1;
    popm(npm+1:npop)=[];
    popc=popc(:);

    newpop=[pop;popc;popm];%新种群
    [newpop,F]=Non_dominate_sort(newpop);
    newpop=Crowd(newpop,F);%计算拥挤度
    newpop=nsga2Sort(newpop);%根据生态等级与拥挤度进行排序
    pop=newpop(1:npop);%只取前npop个个体存活至下一次迭代

    y1=zeros(1,npop);
    y2=zeros(1,npop);
    ys=[pop.y];
    for j=1:npop
        y1(j)=ys(2*j-1);
        y2(j)=ys(2*j);
    end

    %% 绘图
    mean_iteration = [mean_iteration,mean(sum(fall,1))];%这一带的综合值
    iteration = [iteration,it];

    subplot(2,1,1)
    plot(y1,y2,'r*');
    numtitle=num2str(it);
    title('迭代次数=',numtitle);
    xlabel('y1');
    ylabel('y2');
    subplot(2,1,2)
    plot(iteration,mean_iteration);
    ylim([1 2.17]);
    xlabel('iteration');
    ylabel('y1+y2');
    set(gcf,'color','white');
    pause(0.001);
    %frame=getframe(gcf);
    %writeVideo(v,frame);%记录迭代过程

    %% 判断最优
    y_sum = 1*fall(1,:)+1*fall(2,:);%
    y_best(it+1) = max(y_sum);
    if y_best(it+1) == y_best(it)
        n_best = n_best+1;
    else
        n_best = 0;%最优解不一样，则最优解重复次数归0
    end
    it = it+1;
    toc
end
disp(['第',num2str(find(y_sum==y_best(it))),...
            '个个体最优，y_sum=',num2str(y_best(it))]);
%% 参数敏感性分析
% M = nvar*2;
% ns = 100;%采样数
% pointset= sobolset(M);
% R = net(pointset,ns);%生成样本集
% A = R(:,1:nvar);
% B = R(:,nvar+1:end);
% SAB = zeros(ns,nvar,nvar);
% for i=1:nvar
%     A(:,i) = min(x(:,i))+A(:,i).*(max(x(:,i))-min(x(:,i)));%将0~1之间的样本值映射到实际值的范围内
%     B(:,i) = min(x(:,i))+B(:,i).*(max(x(:,i))-min(x(:,i)));
% end
% for i=1:1:nvar
%     tempA = A;
%     tempA(:,i) = B(:,i);
%     SAB(:,:,i) = tempA;
% end
% Y1A = zeros(nvar,1);Y1B = zeros(nvar,1);
% Y1AB = zeros(ns,nvar);
% for i=1:1:nvar
% Y1A = y2_b(c_2,A);
% Y1B = y2_b(c_2,B);
% end
% for i=1:nvar
% Y1AB(:,i) = y2_b(c_2,SAB(:,:,i));
% end
% 
% VarEX = zeros(nvar,1);%一阶影响指数分子
% VarY = var([Y1A;Y1B],1);%分母
% S1 = zeros(nvar,1);%一阶影响指数
% EVarX = zeros(nvar,1);%全局影响指数分子
% ST = zeros(nvar,1);%全局影响指数
% for i=1:nvar
%     for j =1:ns
%         VarEX(i) = VarEX(i)+Y1B(j).*(Y1AB(j,i)-Y1A(j))./ns;
%         EVarX(i) = EVarX(i)+((Y1A(j)-Y1AB(j,i)).^2)./(2.*ns);
%     end
% end
% S1 = VarEX./VarY;
% ST = EVarX./VarY;
% S = [S1 ST];
% figure
% bar(S);

%% 取最优
% y1_all=zeros(1,npop);
% y2_all=zeros(1,npop);
% for i=1:npop
%     y1_all(i)=pop(i).y(1);
%     y2_all(i)=pop(i).y(2);
% end
% y_sum = y1_all +y2_all;
% for i=1:npop
%     if(y_sum(i)==max(y_sum))
%         disp(['第',num2str(i),'个个体最优，y_sum=',num2str(y_sum(i))]);
%         break;
%     end
% end
