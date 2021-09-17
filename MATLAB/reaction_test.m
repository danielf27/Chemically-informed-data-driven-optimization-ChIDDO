function f = reaction_test(t, c, j0_1, j0_2, j0_3, j0_4, alpha_1,alpha_2,alpha_3, alpha_4, E0_1, E0_2,E0_3, E0_4, init_A, init_B, d, volt)

Nx = 20;
dx = d/(Nx+1);
F = 96485; % C/mol
R = 8.314;
T = 300;
D = 5e-9;
elec_dist = 0.01;
z=1;
Pot = volt/elec_dist;
eta_1 = volt-E0_1;
eta_2 = volt-E0_2;
eta_3 = volt-E0_3;
eta_4 = volt-E0_4;
M = D*z*96485*Pot/(R*T);

% 6 species of interest, A, B, C, E, G, H
f = zeros(6*Nx, 1);

f(1) = 0;
f(Nx+1) = 0;
f(2*Nx+1) = 0;
f(3*Nx+1) = 0;
f(4*Nx+1) = 0;
f(5*Nx+1) = 0;

rrate_1 = j0_1*((c(Nx))^1)*(c(2*Nx))*exp((alpha_1*F*eta_1/R/T));
rrate_2 = j0_2*((c(Nx))^2)*((c(2*Nx))^1.15)*exp((alpha_2*F*eta_2/R/T));
rrate_3 = j0_3*((c(Nx))^3)*((c(2*Nx))^1)*exp((alpha_3*F*eta_3/R/T));
rrate_4 = j0_4*((c(2*Nx))^2)*exp((alpha_4*F*eta_4/R/T));

f(Nx) = ((-rrate_1-rrate_2-rrate_3)/(F*dx)) + (D/(dx^2))*(c(Nx-1) - c(Nx)) + (M/(dx))*(c(Nx-1) - c(Nx));
f(2*Nx) = ((-rrate_1-rrate_2-rrate_3-rrate_4)/(F*dx)) + (D/(dx^2))*(c(2*Nx-1) - c(2*Nx)) + (M/(dx))*(c(2*Nx-1) - c(2*Nx));
f(3*Nx) = ((rrate_1)/(F*dx)) + (D/(dx^2))*(c(3*Nx-1) - c(3*Nx)) + (M/(dx))*(c(3*Nx-1) - c(3*Nx));
f(4*Nx) = ((rrate_2)/(F*dx)) + (D/(dx^2))*(c(4*Nx-1) - c(4*Nx)) + (M/(dx))*(c(4*Nx-1) - c(4*Nx));
f(5*Nx) = ((rrate_3)/(F*dx)) + (D/(dx^2))*(c(5*Nx-1) - c(5*Nx)) + (M/(dx))*(c(5*Nx-1) - c(5*Nx));
f(6*Nx) = ((rrate_4)/(F*dx)) + (D/(dx^2))*(c(6*Nx-1) - c(6*Nx)) + (M/(dx))*(c(6*Nx-1) - c(6*Nx));

% Inner points
for j = 2:Nx-1
    f(j) = (D/(dx^2))*(c(j-1) - 2*c(j) + c(j+1)) + (M/(2*dx))*(c(j+1) - c(j-1));
    f(Nx+j) = (D/(dx^2))*(c(Nx+j-1) - 2*c(Nx+j) + c(Nx+j+1)) + (M/(2*dx))*(c(Nx+j+1) - c(Nx+j-1));
    f(2*Nx+j) = (D/(dx^2))*(c(2*Nx+j-1) - 2*c(2*Nx+j) + c(2*Nx+j+1)) + (M/(2*dx))*(c(2*Nx+j+1) - c(2*Nx+j-1));
    f(3*Nx+j) = (D/(dx^2))*(c(3*Nx+j-1) - 2*c(3*Nx+j) + c(3*Nx+j+1)) + (M/(2*dx))*(c(3*Nx+j+1) - c(3*Nx+j-1));
    f(4*Nx+j) = (D/(dx^2))*(c(4*Nx+j-1) - 2*c(4*Nx+j) + c(4*Nx+j+1)) + (M/(2*dx))*(c(4*Nx+j+1) - c(4*Nx+j-1));
    f(5*Nx+j) = (D/(dx^2))*(c(5*Nx+j-1) - 2*c(5*Nx+j) + c(5*Nx+j+1)) + (M/(2*dx))*(c(5*Nx+j+1) - c(5*Nx+j-1));


end