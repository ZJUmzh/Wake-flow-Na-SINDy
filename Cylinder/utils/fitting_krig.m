function An_new = fitting_krig(t_old,t_new,An,n_modes)

addpath('dace')
An(:,n_modes+1:end) = [];
t_old = t_old';
% theta = ones(n_modes,1)*100;
% lob = ones(n_modes,1)*1e-1;
% upb = ones(n_modes,1)*10000;

theta = 100;
lob = 1e-1;
upb = 1000000;

[dmodel,perf] = dacefit(t_old,An,@regpoly0,@corrspline,theta,lob,upb);

[An_new,mse] = predictor(t_new',dmodel);
end