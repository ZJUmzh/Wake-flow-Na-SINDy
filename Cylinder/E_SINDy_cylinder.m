%%%%%%%%%%%%%%%%%%%
% 
% run sims for one noise level and data length to plot ensemble forecasting and UQ
%
%

clear all
close all
clc
addpath('./sparsedynamics/utils')
addpath('./functionfgit/')
% load('E:\study materials\2025\2025summer\matlab_code\dmdc\isdb10.mat');
%% sweep over a set of noise levels and data length to generate heatmap plots
% noise level
eps = 0;
% eps = 0;

% simulation time
tEnd = 1001;

% at each noise level and simulation time, nTest different instantiations of noise are run (model errors and success rate are then averaged for plotting)
nTest1 = 1; % generate models nTest1 times for SINDy
nTest2 = 1; % generate models nTest times for ensemble SINDy


%% hyperparameters
% SINDy sparsifying hyperparameters
lambda = 0.00002;

% ensemble hyperparameters
% data ensembling
nEnsembles = 100; % number of bootstraps (SINDy models using sampled data) in ensemble
ensembleT = 0.6; % Threshold model coefficient inclusion probability: set ensemble SINDy model coefficient to zero if inclusion probability is below ensembleT

% library
nEnsemble1P = 0.9; % percentage of full library that is sampled without replacement for library bagging
nEnsemble2 = 100; % number of bootstraps (SINDy models using sampled library terms) in ensemble
ensT = 0.4; % Threshold library term inclusion probabilty: cut library entries that occur less than ensT

% double bagging
nEnsemblesDD = 100; % number of models in ensemble for data bagging after library bagging


%% common parameters, true Lorenz system, signal power for noise calculation

% generate synthetic Lorenz system data
r = 2;
load PODcoefficients
x = [alpha(1:5001,1:r) alphaS(1:5001,1)];
load PODcoefficients_run1
x1 = [alpha(1:3000,1:r) alphaS(1:3000,1)];

xobs = [x;x1];
xobs= xobs(1:end,:);
x0 = xobs(1,:)';

% x0 = cdata.chemicals.loop1.PV(1);
n = length(x0); 

% set common params
polys = 1:5;
trigs = [];
common_params = {polys,trigs};
gamma = 0;
tol_ode = 1e-6;         % set tolerance (abs and rel) of ode45
options = odeset('RelTol',tol_ode,'AbsTol',tol_ode*ones(1,length(x0)));

% time step
dt = 1;
tspan = 1:dt:size(xobs,1);

% tspan = cdata.chemicals.loop1.t;
% dt = tspan(2)-tspan(1);
%% general parameters

% generate data
% t = tspan;
% x = cdata.chemicals.loop1.PV;

% set rnd number for reproduction
rng(1,'twister')

% add noise
% data before smoothing for plotting
xobsPlotE = xobs;
% build library
% Theta_0 = build_theta(xobs,common_params);
Theta_0 = poolData(xobs,3,5,0);


%% SINDy
% sindy with central difference differentiation
sindy = sindy_cd(xobs,Theta_0,n,lambda,gamma,dt);
%% ENSEMBLES SINDY

%% calculate derivatives
% finite difference differentiation
dxobs_0 = zeros(size(xobs));
dxobs_0(1,:)=(-11/6*xobs(1,:) + 3*xobs(2,:) -3/2*xobs(3,:) + xobs(4,:)/3)/dt;
dxobs_0(2:size(xobs,1)-1,:) = (xobs(3:end,:)-xobs(1:end-2,:))/(2*dt);
dxobs_0(size(xobs,1),:) = (11/6*xobs(end,:) - 3*xobs(end-1,:) + 3/2*xobs(end-2,:) - xobs(end-3,:)/3)/dt;
            
%% Bagging SINDy library
% randomly sample library terms without replacement and throw away terms
% with low inclusion probability
nEnsemble1 = round(nEnsemble1P*size(Theta_0,2));
mOutBS = zeros(nEnsemble1,n,nEnsemble2);
libOutBS = zeros(nEnsemble1,nEnsemble2);
for iii = 1:nEnsemble2
    rs = RandStream('mlfg6331_64','Seed',iii); 
    libOutBS(:,iii) = datasample(rs,1:size(Theta_0,2),nEnsemble1,'Replace',false)';
    mOutBS(:,:,iii) = sparsifyDynamics_n(Theta_0(:,libOutBS(:,iii)),dxobs_0,lambda,n,gamma);
end

inclProbBS = zeros(size(Theta_0,2),n);
for iii = 1:nEnsemble2
    for jjj = 1:n
        for kkk = 1:nEnsemble1
            if mOutBS(kkk,jjj,iii) ~= 0
                inclProbBS(libOutBS(kkk,iii),jjj) = inclProbBS(libOutBS(kkk,iii),jjj) + 1;
            end
        end
    end
end
inclProbBS = inclProbBS/nEnsemble2*size(Theta_0,2)/nEnsemble1;

XiD = zeros(size(Theta_0,2),n);
for iii = 1:n
    libEntry = inclProbBS(:,iii)>ensT;
    XiBias = sparsifyDynamics_n(Theta_0(:,libEntry),dxobs_0(:,iii),lambda,1,gamma);
    XiD(libEntry,iii) = XiBias;
end

                
%% Double bagging SINDy 
% randomly sample library terms without replacement and throw away terms
% with low inclusion probability
% then on smaller library, do bagging

XiDB = zeros(size(Theta_0,2),n);
XiDBmed = zeros(size(Theta_0,2),n);
XiDBs = zeros(size(Theta_0,2),n);
XiDBeOut = zeros(size(Theta_0,2),n,nEnsemblesDD);
inclProbDB = zeros(size(Theta_0,2),n);
for iii = 1:n
    libEntry = inclProbBS(:,iii)>ensT;

    bootstatDD = bootstrp(nEnsemblesDD,@(Theta,dx)sparsifyDynamics_n(Theta,dx,lambda,1,gamma),Theta_0(:,libEntry),dxobs_0(:,iii)); 
    
    XiDBe = [];
    XiDBnz = [];
    for iE = 1:nEnsemblesDD
        XiDBe(:,iE) = reshape(bootstatDD(iE,:),size(Theta_0(:,libEntry),2),1);
        XiDBnz(:,iE) = XiDBe(:,iE)~=0;
        
        XiDBeOut(libEntry,iii,iE) = XiDBe(:,iE);
    end

    % Thresholded bootstrap aggregating (bagging, from bootstrap aggregating)
    XiDBnzM = mean(XiDBnz,2); % mean of non-zero values in ensemble
    inclProbDB(libEntry,iii) = XiDBnzM;
    XiDBnzM(XiDBnzM<ensembleT) = 0; % threshold: set all parameters that have an inclusion probability below threshold to zero

    XiDBmean = mean(XiDBe,2);
    XiDBmedian = median(XiDBe,2);
    XiDBstd = std(XiDBe')';

    XiDBmean(XiDBnzM==0)=0; 
    XiDBmedian(XiDBnzM==0)=0; 
    XiDBstd(XiDBnzM==0)=0; 
    
    XiDB(libEntry,iii) = XiDBmean;
    XiDBmed(libEntry,iii) = XiDBmedian;
    XiDBs(libEntry,iii) = XiDBstd;
    
end
%% compare time series for different ensemble methods
clc
polysIN = 1:2; % skip last rows to oncrease speed, doesnt change results
% skipLastRows = size(XiDB,1)-9; 
skipLastRows = 0;

% sindy
Xi = sindy(1:end-skipLastRows,:);
[tspanSINDy,xSINDY]=ode45(@(t,x)sparseGalerkin_n(t,x,Xi,polysIN),tspan,x0);%,options);  % approximate

% Library bagging 
Xi = XiD(1:end-skipLastRows,:);
[tspanSINDyD,xSINDYXiD]=ode45(@(t,x)sparseGalerkin_n(t,x,Xi,polysIN),tspan,x0);%,options);  % approximate

% Double bagging 
Xi = XiDB(1:end-skipLastRows,:);
[tspanSINDyDB,xSINDYXiDB]=ode45(@(t,x)sparseGalerkin_n(t,x,Xi,polysIN),tspan,x0);%,options);  % approximate

plot(tspan(1:5001),xobs(1:5001,1));
hold on
% plot(tspanSINDy(1:5001),xSINDY(1:5001,1))
% hold on
% plot(tspanSINDyD(1:5001),xSINDYXiD(1:5001,1))
% hold on
plot(tspanSINDyDB(1:5001),xSINDYXiDB(1:5001,1))
corr2(xobs(1:5001,:),xSINDYXiDB(1:5001,:))
xlim([0,5000])
set(gca, 'FontName', 'Times New Roman', 'FontSize', 10);
set(gcf, 'Units', 'centimeters', 'Position', [15,10,14,10]);
set(gcf,'Color',[1 1 1]);
xlabel('Time step')
ylabel('Time coefficient of mode 1')
legend('Simulation','Identified system of E-SINDy');
box on