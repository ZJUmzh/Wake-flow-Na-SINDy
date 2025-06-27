clear
clc
close all

load Wakeflowcoefficients.mat
x = wake_flow_c;

tic
% [6.486344686264463,-3.126624050705976]
% y = respond(6.486344686264463,-3.364186954046732,x);
y = respond(6,-3.36,x);
%%
toc