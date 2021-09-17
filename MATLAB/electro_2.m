function select_4 = electro_2(params, x)

Nx = 20;
n = Nx;

d = 100e-6;
Volt = 1.2;
Init_A = x(1);
Init_B = x(2);

E0_1 = 1;
E0_2 = params(1);
E0_3 = params(2);
E0_4 = 1;

j0_1 = 1e-5;
j0_2 = params(3);
j0_3 = params(4);
j0_4 = 5e-6;

alpha_1 = 0.5;
alpha_2 = params(5);
alpha_3 = params(6);
alpha_4 = 0.5;
D = 5e-9;
z = 1;

% params = readmatrix('Params_2020-06-07_10_alts_cerium.csv');

F = 96485;
R=8.314;
T=300;

Init = zeros(6*Nx,1);
Init(1:Nx) = Init_A*ones(Nx, 1);
Init(Nx+1:2*Nx) = Init_B*ones(Nx, 1);
% times = np.linspace(0,15)
% j0_1, j0_2, j0_3, j0_4, alpha_1,alpha_2,alpha_3, alpha_4, E0_1, E0_2,E0_3, E0_4, init_A, init_B, d, volt
opts = odeset('InitialStep', 1e-5);
[time, conc] = ode15s(@(t,y) reaction_test(t,y,j0_1, j0_2, j0_3, j0_4, alpha_1,alpha_2,alpha_3, alpha_4, E0_1, E0_2,E0_3, E0_4, Init_A, Init_B, d, Volt),[0, 5], Init, opts);
select_4 = conc(end,4*Nx)/(conc(end,3*Nx)+conc(end,4*Nx)+conc(end,5*Nx)+conc(end,6*Nx));


end