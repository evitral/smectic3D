/********************************************
 *                                          *
 *                 pf3d.cpp                 *
 *                                          *
 *     3D Phase Field + FFTW in parallel    *
 *                                          *
 ********************************************/


#include <fftw3-mpi.h>
#include <iostream>
#include <iomanip> // std::setw
#include <fstream>  // read, write
#include <cmath>
#include <cstdlib> // std::exit()
#include <complex>
#include <vector>
#include <cassert> // for assert etc

// Compilation

/*
mpicxx -I /soft/fftw/intel-ompi/3.3-double/include -o pf3d pf3d.cpp
 -L /soft/fftw/intel-ompi/3.3-double/lib -lfftw3_mpi -lfftw3 -lm 
*/


/*****************   MAIN   ****************/


int main (void) {

// ptrdiff_t : integer type, optimizes large transforms 64bit machines

	const ptrdiff_t Nx = 256, Ny = 256, Nz = 256;
	const ptrdiff_t NG = Nx*Ny*Nz;
	ptrdiff_t alloc_local, local_n0, local_0_start;

// FFTW plan

	fftw_plan planPsi, iPlanPsi, planN, iPlanPsiGradx, iPlanPsiGrady, iPlanPsiGradz,
			  iPlanPsiSH, planSx, planSy, planSz, iPlanUx, iPlanUy, iPlanUz;

// Phase Field parameters

	double beta    =  2.0; // 3.0
	double alpha   =  1.0; 
	double epsilon = -0.7;// -1.5
	double gamma   =  1.0;
	double q0      =  1.0;

// Balance of Linear Momentum parameter

	double eta = 1.0;
	double rho = 1.0;

// Parameters, variables and arrays

	int i, i_local, j, k, index, rank, size;

	std::vector <double> Vqy(Ny), Vqz(Nz);
	double scale = 1.0/NG;
	
	const int Nw    = 16;  // number of pts per wavelength

	const double Lx = Nx*2.0*M_PI/(q0*Nw); // Length X
	const double dx = Lx/Nx;
	const double Ly = Ny*2.0*M_PI/(q0*Nw); // Length Y
	const double dy = Ly/Ny;
	const double Lz = Nz*2.0*M_PI/(q0*Nw); // Length Z
	const double dz = Lz/Nz;

	double dt = 0.001;
	double dtd2 = dt/2.0;	

	double L1, sumA, sumA_local, sumB, sumB_local;

	const std::complex<double> iMag(0,1);

	std::complex<double> dotSqVq;

// Initialize MPI

	MPI::Init();
	fftw_mpi_init();

	rank = MPI::COMM_WORLD.Get_rank();
	size = MPI::COMM_WORLD.Get_size();

	if ( rank == 0 ) 
	{
		std::cout << "Using " << size << " processors." << std::endl;
	}

	std::cout << "P(" << rank << ") saying hello." << std::endl;


// Check: np should divide evenly into Nx, Ny and Nz

	if (( Nx%size != 0) || ( Ny%size != 0) || ( Nz%size != 0)) 
	{
		if ( rank == 0) 
		{
			std::cout << "!ERROR! : size =  " << size
			<< " does not divide evenly into Nx, Ny and Nz."
			<< std::endl;
		}
		std::exit(1);
	}

// Allocate local data size

	alloc_local = fftw_mpi_local_size_3d(Nx,Ny,Nz/2+1,MPI::COMM_WORLD,
			                		     &local_n0, &local_0_start);

// Runtime defined vectors

	std::vector<double> Vqx (local_n0);
	
	std::vector<double> psi(size*2*alloc_local);  // suposing they are =
	std::vector<double> psi_local (2*alloc_local);
	std::vector<double> psiNew_local (2*alloc_local);
	std::vector<double> psiSH_local (2*alloc_local);
	std::vector<double> psiGradx_local (2*alloc_local);
	std::vector<double> psiGrady_local (2*alloc_local);
	std::vector<double> psiGradz_local (2*alloc_local);
	std::vector<double> Srx_local (2*alloc_local);
	std::vector<double> Sry_local (2*alloc_local);
	std::vector<double> Srz_local (2*alloc_local);
	std::vector<double> Ux_local (2*alloc_local);
	std::vector<double> Uy_local (2*alloc_local);
	std::vector<double> Uz_local (2*alloc_local);
	std::vector<double> Nr_local (2*alloc_local);

	std::vector< std::complex<double> > psiq_local (alloc_local);
	std::vector< std::complex<double> > psiqNew_local (alloc_local);
	std::vector< std::complex<double> > psiqGradx_local (alloc_local);
	std::vector< std::complex<double> > psiqGrady_local (alloc_local);
	std::vector< std::complex<double> > psiqGradz_local (alloc_local);
	std::vector< std::complex<double> > psiqSH_local (alloc_local);
	std::vector< std::complex<double> > Uqx_local (alloc_local);
	std::vector< std::complex<double> > Uqy_local (alloc_local);
	std::vector< std::complex<double> > Uqz_local (alloc_local);
	std::vector< std::complex<double> > Sqx_local (alloc_local);
	std::vector< std::complex<double> > Sqy_local (alloc_local);
	std::vector< std::complex<double> > Sqz_local (alloc_local);
	std::vector< std::complex<double> > Nq_local (alloc_local);
	std::vector< std::complex<double> > NqPast_local (alloc_local);

	std::vector< std::complex<double> > opSH (alloc_local);
	std::vector< std::complex<double> > aLin (alloc_local);
	std::vector< std::complex<double> > C1 (alloc_local);
	std::vector< std::complex<double> > C2 (alloc_local);
	std::vector< std::complex<double> > mq2 (alloc_local);
	std::vector< std::complex<double> > CM1 (alloc_local);
	std::vector< std::complex<double> > CM2x (alloc_local);
	std::vector< std::complex<double> > CM2y (alloc_local);
	std::vector< std::complex<double> > CM2z (alloc_local);

// Create Fourier mode vectors 

	for ( i_local = 0; i_local < local_n0; i_local++ ) 
	{	
		i = i_local + local_0_start;

		if ( i < Nx/2 + 1 ) 
		{
	 		Vqx[i_local] = 2.0*M_PI*i/(dx*Nx);
		} 
		else 
		{	
			Vqx[i_local] = -(Nx-i)*2.0*M_PI/(dx*Nx);
		}	
	}	

	Vqy[0] = 0; Vqy[Ny/2] = 1.0*M_PI/dy;

	for ( j = 1; j < Ny/2; j++ ) 
	{
    	Vqy[j] = 2.0*M_PI*j/(dy*Ny);
    	Vqy[Ny/2+j] = -(Ny/2-j)*2.0*M_PI/(dy*Ny);
	}

	Vqz[0] = 0; Vqz[Nz/2] = 1.0*M_PI/dz;

	for ( k = 1; k < Nz/2; k++ ) 
	{
		Vqz[k] = 2.0*M_PI*k/(dz*Nz);
		Vqz[Nz/2+k] = -(Nz/2-k)*2.0*M_PI/(dz*Nz);
	}

// Define Swift Hohenberg linear operator alongside
// Crank-Nicolson + Adams-Bashforth scheme constants in FS

	double q02 = q0*q0;

	for ( i_local = 0; i_local < local_n0; i_local++ )
	{
		i = i_local + local_0_start;

		for ( j = 0; j < Ny; j++ ) {
		for ( k = 0; k < Nz/2+1; k++ )
		{
			index = (i_local*Ny + j) * (Nz/2+1) + k;
			opSH[index] = pow((pow(Vqx[i_local],2)+pow(Vqy[j],2)+pow(Vqz[k],2)-q02),2);
			aLin[index] = epsilon - opSH[index];
		//	aLin[index] = epsilon - pow((pow(Vqx[i],2) + pow(Vqy[j],2)
		//				+ pow(Vqz[k],2) - q02),2);
			mq2[index] = pow(Vqx[i_local],2)+pow(Vqy[j],2)+pow(Vqz[k],2);
			CM1[index] = 1.0/(rho*eta*mq2[index]);
			CM2x[index] = Vqx[i_local]/mq2[index]; 
			CM2y[index] = Vqy[j]/mq2[index];
			CM2z[index] = Vqz[k]/mq2[index];
			C1[index] = 1.0+dtd2*aLin[index];
			C2[index] = 1.0-dtd2*aLin[index];
		}}
	}

	CM1[0]  = 0.0;
	CM2x[0] = 0.0;
	CM2y[0] = 0.0;
	CM2z[0] = 0.0;

// Create forward FFTW plan

	planPsi = fftw_mpi_plan_dft_r2c_3d(Nx,Ny,Nz,
	            	    psi_local.data(),
				        reinterpret_cast<fftw_complex*>(psiq_local.data()),
				        MPI::COMM_WORLD,FFTW_MEASURE);

	planSx = fftw_mpi_plan_dft_r2c_3d(Nx,Ny,Nz,
	            	    Srx_local.data(),
				        reinterpret_cast<fftw_complex*>(Sqx_local.data()),
				        MPI::COMM_WORLD,FFTW_MEASURE);
	
	planSy = fftw_mpi_plan_dft_r2c_3d(Nx,Ny,Nz,
	            	    Sry_local.data(),
				        reinterpret_cast<fftw_complex*>(Sqy_local.data()),
				        MPI::COMM_WORLD,FFTW_MEASURE);

	planSz = fftw_mpi_plan_dft_r2c_3d(Nx,Ny,Nz,
	            	    Srz_local.data(),
				        reinterpret_cast<fftw_complex*>(Sqz_local.data()),
				        MPI::COMM_WORLD,FFTW_MEASURE);

	planN = fftw_mpi_plan_dft_r2c_3d(Nx,Ny,Nz,
	            	    Nr_local.data(),
				        reinterpret_cast<fftw_complex*>(Nq_local.data()),
				        MPI::COMM_WORLD,FFTW_MEASURE);

// Create backward FFTW plan

	iPlanPsi = fftw_mpi_plan_dft_c2r_3d(Nx,Ny,Nz,
	             		reinterpret_cast<fftw_complex*>(psiqNew_local.data()),
				        psiNew_local.data(),
				        MPI::COMM_WORLD,FFTW_MEASURE);

	iPlanPsiGradx = fftw_mpi_plan_dft_c2r_3d(Nx,Ny,Nz,
	             		reinterpret_cast<fftw_complex*>(psiqGradx_local.data()),
				        psiGradx_local.data(),
				        MPI::COMM_WORLD,FFTW_MEASURE);

	iPlanPsiGrady = fftw_mpi_plan_dft_c2r_3d(Nx,Ny,Nz,
	             		reinterpret_cast<fftw_complex*>(psiqGrady_local.data()),
				        psiGrady_local.data(),
				        MPI::COMM_WORLD,FFTW_MEASURE);

	iPlanPsiGradz = fftw_mpi_plan_dft_c2r_3d(Nx,Ny,Nz,
	             		reinterpret_cast<fftw_complex*>(psiqGradz_local.data()),
				        psiGradz_local.data(),
				        MPI::COMM_WORLD,FFTW_MEASURE);

	iPlanPsiSH = fftw_mpi_plan_dft_c2r_3d(Nx,Ny,Nz,
	             		reinterpret_cast<fftw_complex*>(psiqSH_local.data()),
				        psiSH_local.data(),
				        MPI::COMM_WORLD,FFTW_MEASURE);

	iPlanUx = fftw_mpi_plan_dft_c2r_3d(Nx,Ny,Nz,
	             		reinterpret_cast<fftw_complex*>(Uqx_local.data()),
				        Ux_local.data(),
				        MPI::COMM_WORLD,FFTW_MEASURE);

	iPlanUy = fftw_mpi_plan_dft_c2r_3d(Nx,Ny,Nz,
	             		reinterpret_cast<fftw_complex*>(Uqy_local.data()),
				        Uy_local.data(),
				        MPI::COMM_WORLD,FFTW_MEASURE);

	iPlanUz = fftw_mpi_plan_dft_c2r_3d(Nx,Ny,Nz,
	             		reinterpret_cast<fftw_complex*>(Uqz_local.data()),
				        Uz_local.data(),
				        MPI::COMM_WORLD,FFTW_MEASURE);

// Inform allocated size for each rank

	std::cout << "Rank "<< rank << std::endl 
	<< "alloc_local: " << alloc_local <<  std::endl
	<< "local_n0: " << local_n0 << std::endl
	<< "local_0_start: " << local_0_start << std::endl;

// Initial condition

	double Amp = 1.2441;
	double Qi  = 0.5;

	for ( i_local = 0; i_local < local_n0; i_local++ ) 
	{
		i = i_local + local_0_start;

		for ( j = 0; j < Ny; j++ ) {
		for ( k = 0; k < Nz; k++ ) 
		{	
			index = (i_local*Ny + j) * (2*(Nz/2+1)) + k;
			if ( (k > 24) && (k < 232) ) // 18 110
			{
			//	psi_local[index] = cos(q0*k*dz);
				psi_local[index] = Amp*cos(q0*k*dz)
								 + Amp*0.1*sin(q0*k*dz)*(cos(Qi*i*dx)+cos(Qi*j*dy));
			}
			else
			{
				psi_local[index] = 0.0;
			}
		}}
	}

	MPI::COMM_WORLD.Barrier();

/*
	for ( i_local = 0; i_local < local_n0; i_local++ )
	{
		std::cout << "Rank " << rank << " : " << Vqx[i_local] << std::endl;
	}
*/


	MPI::COMM_WORLD.Gather(psi_local.data(),2*alloc_local,MPI::DOUBLE,
	         		       psi.data(),2*alloc_local, MPI::DOUBLE,0);

	std::ofstream L1_output;

	if (rank == 0 )
	{
		std::ofstream psiIC_output("psiIC.dat");
		assert(psiIC_output.is_open());

		for ( i = 0; i < Nx; i++ ) {
		for ( j = 0; j < Ny; j++ ) {
		for ( k = 0; k < Nz; k++ ) 
		{
			index = (i*Ny + j) * (2*(Nz/2+1)) + k;	
			psiIC_output << psi[index] << "\n ";
		}}}

		psiIC_output.close();

		std::ofstream L1_output("L1.dat");
		assert(L1_output.is_open());
		L1_output.close();
	}

// First NqPast

	for ( i_local = 0; i_local < local_n0; i_local++ ) {
	for ( j = 0; j < Ny; j++ ) {
	for ( k = 0; k < Nz; k++ )
	{
		index = (i_local*Ny + j) * (2*(Nz/2+1)) + k;
		Nr_local[index] = beta*pow(psi_local[index],3) 
						- gamma*pow(psi_local[index],5);
	}}}

	fftw_execute(planN);

	NqPast_local = Nq_local;


// Main loop

	sleep(5);

	int countL1 = 0;

	int countSave = 0;

	L1 = 1.0;

	double limL1 = pow(10.0,-6);


	while ( L1 > limL1) 
	{

		countL1++;

		fftw_execute(planPsi);

// Obtain the spectral gradient of psi and move to Real Space

		for ( i_local = 0; i_local < local_n0; i_local++ ) {
		for ( j = 0; j < Ny; j++ ) {
		for ( k = 0; k < Nz/2+1; k++ )
		{
			index = (i_local*Ny + j) * (Nz/2+1) + k;
			psiqGradx_local[index] = scale*iMag*Vqx[i_local]*psiq_local[index];
			psiqGrady_local[index] = scale*iMag*Vqy[j]*psiq_local[index];
			psiqGradz_local[index] = scale*iMag*Vqz[k]*psiq_local[index];
			psiqSH_local[index] = scale*opSH[index]*psiq_local[index];
		}}}
		
		fftw_execute(iPlanPsiGradx);
		fftw_execute(iPlanPsiGrady);
		fftw_execute(iPlanPsiGradz);
		fftw_execute(iPlanPsiSH);

// Compute the divergence of the stress and move to Fourier Space

		for ( i_local = 0; i_local < local_n0; i_local++ ) {
		for ( j = 0; j < Ny; j++ ) {
		for ( k = 0; k < Nz; k++ )
		{
			index = (i_local*Ny + j) * (2*(Nz/2+1)) + k;
			Srx_local[index] = psiSH_local[index]*psiGradx_local[index];
			Sry_local[index] = psiSH_local[index]*psiGrady_local[index];
			Srz_local[index] = psiSH_local[index]*psiGradz_local[index];
		}}}
		
		fftw_execute(planSx);
		fftw_execute(planSy);
		fftw_execute(planSz);

// Compute velocity in Fourier Space and transform back

		for ( i_local = 0; i_local < local_n0; i_local++ ) {
		for ( j = 0; j < Ny; j++ ) {
		for ( k = 0; k < Nz/2+1; k++ )
		{
			index = (i_local*Ny + j) * (Nz/2+1) + k;
			dotSqVq = Vqx[i_local]*Sqx_local[index]
				    + Vqy[j]*Sqy_local[index]
				    + Vqz[k]*Sqz_local[index]; 
			Uqx_local[index] = scale*CM1[index]*(Sqx_local[index]
							 - CM2x[index]*dotSqVq);
			Uqy_local[index] = scale*CM1[index]*(Sqy_local[index]
							 - CM2y[index]*dotSqVq);
			Uqz_local[index] = scale*CM1[index]*(Sqz_local[index]
							 - CM2z[index]*dotSqVq);
		}}}

		fftw_execute(iPlanUx);
		fftw_execute(iPlanUy);
		fftw_execute(iPlanUz);
	
// Nonlinear + advective terms: Real to Fourier Space
		
		for ( i_local = 0; i_local < local_n0; i_local++ ) {
		for ( j = 0; j < Ny; j++ ) {
		for ( k = 0; k < Nz; k++ )
		{
			index = (i_local*Ny + j) * (2*(Nz/2+1)) + k;
			Nr_local[index] = beta*pow(psi_local[index],3) 
							- gamma*pow(psi_local[index],5)
							- Ux_local[index]*psiGradx_local[index]
							- Uy_local[index]*psiGrady_local[index]
							- Uz_local[index]*psiGradz_local[index];
		}}}
	
		fftw_execute(planN);

// Compute New psi in Fourier Space and transform back

		for ( i_local = 0; i_local < local_n0; i_local++ ) {
		for ( j = 0; j < Ny; j++ ) {
		for ( k = 0; k < Nz/2+1; k++ )
		{
			index = (i_local*Ny + j) * (Nz/2+1) + k;
			psiqNew_local[index] = scale*(C1[index]*psiq_local[index] 
					   		 + dtd2*(3*Nq_local[index]-NqPast_local[index])) 
					         / C2[index];
		}}}

		fftw_execute(iPlanPsi);

// Present spectral nonlinear + advective terms becomes Old

		NqPast_local = Nq_local;

// Calculate L1 (under count condition)

		if ( countL1 == 100 )
		{
			sumA_local = 0.0; sumB_local = 0.0;
			sumA = 0.0;       sumB = 0.0;

			for ( i_local = 0; i_local < local_n0; i_local++ ) {
			for ( j = 0; j < Ny; j++ ) {
			for ( k = 0; k < Nz; k++ )
			{
				index = (i_local*Ny + j) * (2*(Nz/2+1)) + k;
				sumA_local = sumA_local  
						   + fabs(psiNew_local[index] - psi_local[index]);
				sumB_local = sumB_local + fabs(psiNew_local[index]);
			}}}

			MPI::COMM_WORLD.Reduce(&sumA_local,&sumA,1,MPI::DOUBLE,MPI::SUM,0);
			MPI::COMM_WORLD.Reduce(&sumB_local,&sumB,1,MPI::DOUBLE,MPI::SUM,0);

			if ( rank == 0)
			{
				L1 = sumA/(dt*sumB);
				L1_output.open("L1.dat",std::ios_base::app); // append result
     			assert(L1_output.is_open());
      			L1_output << L1 << "\n";
      			L1_output.close();
			}

			MPI::COMM_WORLD.Bcast(&L1,1,MPI::DOUBLE,0);

			countL1 = 0;

			countSave++;

// Save psi (under count condition)

			if ( countSave == 100 )
			{
				MPI::COMM_WORLD.Gather(psiNew_local.data(),2*alloc_local,MPI::DOUBLE,
	         		       psi.data(),2*alloc_local, MPI::DOUBLE,0);

				if (rank == 0 )
				{
					std::ofstream psi_output("psi.dat");
					assert(psi_output.is_open());

					for ( i = 0; i < Nx; i++ ) {
					for ( j = 0; j < Ny; j++ ) {
					for ( k = 0; k < Nz; k++ ) 
					{
						index = (i*Ny + j) * (2*(Nz/2+1)) + k;
						psi_output << psi[index] << "\n ";
					}}}

					psi_output.close();
				}

				countSave = 0;
			}			
		}

		psi_local = psiNew_local;
	}

	MPI::COMM_WORLD.Barrier();

	MPI::COMM_WORLD.Gather(psi_local.data(),2*alloc_local,MPI::DOUBLE,
	         		       psi.data(),2*alloc_local, MPI::DOUBLE,0);

	if (rank == 0 )
	{
		std::ofstream psi_output("psi.dat");
		assert(psi_output.is_open());

		for ( i = 0; i < Nx; i++ ) {
		for ( j = 0; j < Ny; j++ ) {
		for ( k = 0; k < Nz; k++ ) 
		{
			index = (i*Ny + j) * (2*(Nz/2+1)) + k;
			psi_output << psi[index] << "\n ";
		}}}

		psi_output.close();

		std::cout << std::endl << "... and, that's a wrap." << std::endl;
	}

	fftw_destroy_plan(planPsi);
	fftw_destroy_plan(planSx);
	fftw_destroy_plan(planSy);
	fftw_destroy_plan(planSz);
	fftw_destroy_plan(planN);
	fftw_destroy_plan(iPlanPsi);
	fftw_destroy_plan(iPlanUx);
	fftw_destroy_plan(iPlanUy);
	fftw_destroy_plan(iPlanUz);
	fftw_destroy_plan(iPlanPsiSH);
	fftw_destroy_plan(iPlanPsiGradx);
	fftw_destroy_plan(iPlanPsiGrady);
	fftw_destroy_plan(iPlanPsiGradz);

	MPI::Finalize();

}
