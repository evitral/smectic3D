%--------------------------------------%
%                                      %
%	       energyPsi.m                 %
%                                      %
%  Compute the Lyapunov functional     %
%  associated for small perturbations  %
%                                      %
%--------------------------------------%

clear all; 

psiData = importdata('psi512.dat');

%% Spatial parameters

np = 512;
Nw = 16.0;
q0 = 1.0;
q02 = q0*q0;
q04 = q02*q02;

L = np*2.0*pi()/(q0*Nw);
dx = L/np;
dq = 2*pi()/L;

[x,y,z] = meshgrid(0:dx:L-dx);
xV = 0:dx:(L-dx);


%% Lyapunov functional parameters

epsilonD2 = -0.7/2;
alphaD2   =  1.0/2;
betaD4    =  2.0/4;
gammaD6   =  1.0/6;

Amp = 1.2441;

%% Organize psi0

psi0 = zeros(np,np,np);

for i = 1:np
for j = 1:np
for k = 1:np
    psi0(i,j,k) = psiData(np*np*(i-1)+np*(j-1)+k);
end
end
end


%% Define containers

psi = zeros(np,np,np);
dpsiz = zeros(np,np,np);
energyT = zeros(1,30);
energyD = zeros(np,np,np);


%% Define finite difference 1/d

dx4 = 1.0/(dx.^4);
dy4 = 1.0/(dx.^4);
dz4 = 1.0/(dx.^4);

dx2y2 = 1.0/(dx.^4);
dx2z2 = 1.0/(dx.^4);
dy2z2 = 1.0/(dx.^4);

dx2 = 1.0/(dx.^2);
dy2 = 1.0/(dx.^2);
dz2 = 1.0/(dx.^2);


%% Compute psi0 z-derivative

for i = 1:np
for j = 1:np
for k = 1:np

    if k == np
        kp = 1; kpp = 2;
    elseif k == np-1
        kp = np; kpp = 1;
    else
        kp = k+1; kpp = k+2;
    end

    if k == 1
        km = np; kmm = np-1;
    elseif k == 2
        km = 1; kmm = np;
    else
        km = k-1; kmm = k-2;
    end

    dpsiz(i,j,k) = - (1.0/12)*psi0(i,j,kpp) + (2/3)*psi0(i,j,kp) ...
                   - (2/3)*psi0(i,j,kmm) + (1/12)*psi0(i,j,km);
end
end
end


%% Compute energy

index = 0;

for Qi  = -dq:dq:1.4
    
    index = index + 1;    

    if index == 1
        psi = psi0;
    else
    
		for i = 1:np
    	for j = 1:np
    	    %for k = 25:np-24
    	    for k = 1:np
    	    %psi(i,j,k) = V(i,j,k) + Amp*0.1*sin(q0*k*dx)*(cos(Qi*i*dx));
    	    %psi(i,j,k) = psi0(i,j,k) + 0.01*dpsiz(i,j,k)*Qi*k*dx;
    	    psi(i,j,k) = psi0(i,j,k) + dpsiz(i,j,k)*0.01*cos((1+Qi)*k*dx);
    	    end
    	end
    	end

    end

%    psi2 = psi.^2;

    for i = 1:np
    for j = 1:np
    for k = 1:np

    if i == 1
        im = np; imm = np-1;
    elseif i == 2
    	im = 1; imm = np;
    else
	im = i-1; imm = i-2;
    end

    if j == 1
        jm = np; jmm = np-1;
    elseif j == 2
        jm = 1; jmm = np;
    else
        jm = j-1; jmm = j-2;
    end

    if k == 1
        km = np; kmm = np-1;
    elseif k == 2
        km = 1; kmm = np;
    else
        km = k-1; kmm = k-2;
    end

    if i == np
        ip = 1; ipp = 2;
    elseif i == np-1
        ip = np; ipp = 1;
    else
        ip = i+1; ipp = i+2;
    end

    if j == np
        jp = 1; jpp = 2;
    elseif j == np-1
        jp = np; jpp = 1;
    else
        jp = j+1; jpp = j+2;
    end

    if k == np
        kp = 1; kpp = 2;
    elseif k == np-1
        kp = np; kpp = 1;
    else
        kp = k+1; kpp = k+2;
    end

  
%{
  
    dpx4 = dx4*(psi2(imm,j,k) - 4*psi2(im,j,k) + 6*psi2(i,j,k) ... 
         - 4*psi2(ip,j,k) + psi2(ipp,j,k));
    dpy4 = dy4*(psi2(i,jmm,k) - 4*psi2(i,jm,k) + 6*psi2(i,j,k) ...
         - 4*psi2(i,jp,k) + psi2(i,jpp,k));
    dpz4 = dz4*(psi2(i,j,kmm) - 4*psi2(i,j,km) + 6*psi2(i,j,k) ...
         - 4*psi2(i,j,kp) + psi2(i,j,kpp));

    dpx2y2 = dx2y2*(psi2(ip,jp,k) + psi2(ip,jm,k) - 2*(psi2(ip,j,k) ... 
           + psi2(i,jp,k) - 2*psi2(i,j,k) + psi2(i,jm,k) + psi2(im,j,k))... 
           + psi2(im,jp,k) + psi2(im,jm,k));

    dpy2z2 = dy2z2*(psi2(i,jp,kp) + psi2(i,jp,km) - 2*(psi2(i,jp,k) ... 
           + psi2(i,j,kp) - 2*psi2(i,j,k) + psi2(i,j,km) + psi2(i,jm,k))...
           + psi2(i,jm,kp) + psi2(i,jm,km));

    dpx2z2 = dx2z2*(psi2(ip,j,kp) + psi2(ip,j,km) - 2*(psi2(ip,j,k) ...
           + psi2(i,j,kp) - 2*psi2(i,j,k) + psi2(i,j,km) + psi2(im,j,k))...
           + psi2(im,j,kp) + psi2(im,j,km));

    dpz2  = dz2*(psi2(i,j,km) - 2*psi2(i,j,k) + psi2(i,j,kp));
    dpy2  = dy2*(psi2(i,jm,k) - 2*psi2(i,j,k) + psi2(i,jp,k));
    dpx2  = dx2*(psi2(im,j,k) - 2*psi2(i,j,k) + psi2(ip,j,k));

    energyD(i,j,k) = ...
    - epsilonD2*psi2(i,j,k) ...
    + alphaD2*(q04*psi2(i,j,k) + 2*q02*(dpx2 + dpy2 + dpz2) ...
    + dpx4 + dpy4 + dpz4 + 2*(dpx2y2 + dpy2z2 + dpx2z2)) ...
    - betaD4*psi(i,j,k).^4 + gammaD6*psi(i,j,k).^6;

%}

    dpz2  = dz2*(psi(i,j,km) - 2*psi(i,j,k) + psi(i,j,kp));
    dpy2  = dy2*(psi(i,jm,k) - 2*psi(i,j,k) + psi(i,jp,k));
    dpx2  = dx2*(psi(im,j,k) - 2*psi(i,j,k) + psi(ip,j,k));

	psi2 = psi(i,j,k)*psi(i,j,k);
	psi4 = psi2*psi2;
	psi6 = psi4*psi2;

    energyD(i,j,k) = ...
    - epsilonD2*psi2 ...
    + alphaD2*(q02*psi(i,j,k) + dpx2 + dpy2 + dpz2).^2 ...
    - betaD4*psi4 + gammaD6*psi6;

    end
    end
    end

    energyT(index) = trapz(xV,trapz(xV,trapz(xV,energyD,3),2));

	save energyT2.mat energyT

end


