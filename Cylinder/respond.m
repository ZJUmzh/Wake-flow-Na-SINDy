function y=respond(k,lambda,x,x1)
tic
addpath('./utils');
addpath('./data');
% 1.014985851275355,-6.756790293423949
k = round(k);
lambda = 10^(lambda);
dt = 0.02;
r=2;
t_old = 0:0.02:100;
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
%% load data from second run and compute derivative
x1_new = zeros((size(x1,1)-1)*k+1,M);
x1_new(1:k:end,:) = x1;
for i = 2:k
    x1_new(i:k:end,:) = x1_new(1:k:end-k,:)+(i-1)/k*(x1_new(k+1:k:end,:)-x1_new(1:k:end-k,:));
end
% compute Derivative
x_der = diag(1*ones(1,size(x1_new,1)-2),2)+diag(-8*ones(1,size(x1_new,1)-1),1)+diag(8*ones(1,size(x1_new,1)-1),-1)+diag(-1*ones(1,size(x1_new,1)-2),-2);
x_der = sparse(x_der);
dx1 = x1_new'*x_der/(12*dt);
dx1(:,1) = (x1_new(2,:)-x1_new(1,:))/dt;
dx1(:,2) = (x1_new(3,:)-x1_new(1,:))/(2*dt);
dx1(:,end) = (x1_new(end,:)-x_new(end-1,:))/dt;
dx1(:,end-1) = (x1_new(end,:)-x1_new(end-2,:))/(2*dt);
dx1 = dx1';
%% concatenate
x_new = [x_new; x1_new];
dx = [dx; dx1];
%% pool Data
Theta = poolData(x_new,M,polyorder,usesine);
m = size(Theta,2);
dx = dx(1:end,:);
Xi = sparsifyDynamics(Theta,dx,lambda,3);
% note that there are constant order terms..
%% calculate
tspan = t_new;
options = odeset('RelTol',1e-8,'AbsTol',1e-8*ones(1,r+1));
x0 = x_new(1,:);%
[tD,xD]=ode45(@(t,x)sparseGalerkin(t,x,Xi,polyorder,usesine),tspan,x0,options);
if size(xD,1)<length(t_new)
    y = [0,0];
else
    % y = corr2(x(:,1),xD(1:k:end,1))+0*corr2(x(:,2),xD(1:k:end,2))+0*corr2(x(:,3),xD(1:k:end,3));
    y = corr2(x(:,:),xD(1:k:end,:));
    y = [y,1/k];
end
toc
end