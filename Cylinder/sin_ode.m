clear
clc
close all

tspan = 0:0.01:pi;
x0 = 0;

options = odeset('RelTol',1e-8,'AbsTol',1e-8);
[tD,xD]=ode45(@(t,x) sin_o(t,x),tspan,x0,options);

plot(tD,xD);
dxD = real(gradient(xD))
function dx = sin_o(t,x)
    dx = real(sqrt(1-x^2));
    if abs(real(dx))<=0.01
        dx = -dx;
    end
end