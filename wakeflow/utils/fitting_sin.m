function An_new = fitting_sin(t_old,t_new,An,n_modes)

for i = 1:n_modes
    a_fit = fit(t_old',An(:,i),'sin8','Normalize','on');
    An_new(:,i) = a_fit(t_new);
end



end