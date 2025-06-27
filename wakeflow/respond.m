function y=respond(k,lambda,x)
addpath('./utils');
addpath('./data');
k = round(k);
lambda = 10^(lambda);
dt = 1;
r=2;
t_old = 0:(size(x,1)-1);
dt = dt/k;
t_new = zeros(1,(length(t_old)-1)*k+1);
t_new(1:k:end) = t_old;
polyorder = 5;
usesine = 0;

%% load data from first run and compute derivative
M = size(x,2);
x_new = zeros((size(x,1)-1)*k+1,M);
x_new(1:k:end,:) = x;
% compute Derivative
eps = 0;
% 
for i = 2:k
    x_new(i:k:end,:) = x_new(1:k:end-k,:)+(i-1)/k*(x_new(k+1:k:end,:)-x_new(1:k:end-k,:));
    t_new(i:k:end) = t_new(1:k:end-k)+(i-1)/k*(t_new(k+1:k:end)-t_new(1:k:end-k)); 
end

x_der = diag(1*ones(1,size(x_new,1)-2),2)+diag(-8*ones(1,size(x_new,1)-1),1)+diag(8*ones(1,size(x_new,1)-1),-1)+diag(-1*ones(1,size(x_new,1)-2),-2);
x_der = sparse(x_der);

dx = x_new'*x_der/(12*dt);
dx(:,1) = (x_new(2,:)-x_new(1,:))/dt;
dx(:,2) = (x_new(3,:)-x_new(1,:))/(2*dt);
dx(:,end) = (x_new(end,:)-x_new(end-1,:))/dt;
dx(:,end-1) = (x_new(end,:)-x_new(end-2,:))/(2*dt);
% 
dx = dx';
%% pool Data
Theta = poolData(x_new,M,polyorder,usesine);
m = size(Theta,2);
dx = dx(1:end,:);
Xi = sparsifyDynamics(Theta,dx,lambda,3);
% note that there are constant order terms..
%% calculate
global startTime maxTime;
startTime = tic;
maxTime = 5;
tspan = t_new;
options = odeset('RelTol',1e-8,'AbsTol',1e-8*ones(1,r+1),'Events',@myEventsFcn);
x0 = x_new(1,:);%
% try
%     [tD,xD]=ode45(@(t,x)sparseGalerkin(t,x,Xi,polyorder,usesine),tspan,x0,options);
% catch ME
%     if strcmp(ME.identifier,'MATLAB:ode45:Timeout')
%         disp('ODE 求解超时，停止求解');
%     else
%         rethrow(ME);
%     end
% end

[tD,xD,te,ye,ie]=ode45(@(t,x)sparseGalerkin(t,x,Xi,polyorder,usesine),tspan,x0,options);
if ~isempty(te)
    fprintf('ODE 求解在 t = %.2f 时因超时停止。\n', te(end));
    % 返回特定结果0
else
    fprintf('ODE 求解正常完成。\n');
    % 返回正常结果
end
if size(xD,1)<length(t_new)
    y = [0,0];
else
    % y = corr2(x(:,1),xD(1:k:end,1))+0*corr2(x(:,2),xD(1:k:end,2))+0*corr2(x(:,3),xD(1:k:end,3));
    % y = corr2(x(:,:),xD(1:k:end,:));
    R2 = 1- sum((x-xD(1:k:end,:)).^2,"all")./sum((x-mean2(x)).^2,"all")
    y = R2;
    y = [y,1/k];
end
end
function [value, isterminal, direction] = myEventsFcn(t, y)
    global startTime maxTime;
    elapsedTime = toc(startTime);
    value = elapsedTime - maxTime; % 当运行时间超过maxTime时，value变为正值
    isterminal = 1; % 1表示满足事件条件时终止积分
    direction = 1; % 1表示仅当value从负变正时触发事件
end