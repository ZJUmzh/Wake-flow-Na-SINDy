clear
clc
close all

r = 2;
load PODcoefficients
x = [alpha(1:5001,1:r) alphaS(1:5001,1)];
load PODcoefficients_run1
x1 = [alpha(1:3000,1:r) alphaS(1:3000,1)];

tic
% y = respond(1.523471465362219,-2.977649143761303,x,x1);
% 1.074764251460167,-6.793023515561676
y = respond(1,-6.744729445620204,x,x1);
%%
toc