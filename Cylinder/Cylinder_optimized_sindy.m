clear
clc
close all

addpath('./utils');
addpath('./data');

dt = 0.02;
r=2;
t_old = 0:0.02:100;
k = 1;
dt = dt/k;
t_new = zeros(1,(length(t_old)-1)*k+1);
t_new(1:k:end) = t_old;
polyorder = 5;
usesine = 0;

%% load data from first run and compute derivative
load PODcoefficients
x = [alpha(1:5001,1:r) alphaS(1:5001,1)];
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
load PODcoefficients_run1
x1 = [alpha(1:3000,1:r) alphaS(1:3000,1)];
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

lambda = 10^(-6.744729445620204);
Xi = sparsifyDynamics(Theta,dx,lambda,3);
% note that there are constant order terms... this is because fixed points
% are not at (0,0,0) for this data
% poolDataLIST({'x','y','z'},Xi,M,polyorder,usesine);

%% - second figure: initial portion
figure(1)

tspan = t_new;
options = odeset('RelTol',1e-8,'AbsTol',1e-8*ones(1,r+1));
x0 = x_new(1,:);%
[tD,xD]=ode45(@(t,x)sparseGalerkin(t,x,Xi,polyorder,usesine),tspan,x0,options);
%
subplot(1,2,1)
color_line3(x(:,1),x(:,2),x(:,end),.02*(1:length(x(:,end))),'LineWidth',1.5);
view(27,16)
grid on
xlabel('x')
ylabel('y')
zlabel('z')
axis([-200 200 -200 200 -160 20])
set(gca, 'FontName', 'Times New Roman', 'FontSize', 10);
box on
subplot(1,2,2)
dtD = [0; diff(tD)];
color_line3(xD(:,1),xD(:,2),xD(:,end),.02*(1:length(xD(:,end))),'LineWidth',1.5);
view(27,16)
grid on
xlabel('x')
ylabel('y')
zlabel('z')
axis([-200 200 -200 200 -160 20])

set(gca, 'FontName', 'Times New Roman', 'FontSize', 10);
set(gcf, 'Units', 'centimeters', 'Position', [15,10,14,5]);
set(gcf,'Color',[1 1 1]);
% set(gca,'LineWidth',2);
box on

%%
figure(2)
plot(t_old,x(:,1));
hold on
plot(t_new,xD(:,1));
corr2(x(:,:),xD(1:k:end,:))
xlabel('Time step')
ylabel('Time coefficient of mode 1')

set(gca, 'FontName', 'Times New Roman', 'FontSize', 10);
set(gcf, 'Units', 'centimeters', 'Position', [15,10,14,10]);
set(gcf,'Color',[1 1 1]);
legend('Simulation','Identified system of Na-SINDy');
box on
% x00 = find(x(1:5000,1).*x(2:5001,1)<0);