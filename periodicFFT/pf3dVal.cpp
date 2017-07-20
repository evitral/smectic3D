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

	const ptrdiff_t Nx = 512, Ny = 512, Nz = 512;
	const ptrdiff_t NG = Nx*Ny*Nz;
	ptrdiff_t alloc_local, local_n0, local_0_start;

// FFTW plan

	fftw_plan planPsi, iPlanPsi, planN;

// Phase Field parameters

	double beta    =  1.0; // 3.0
	double alpha   =  1.0; 
	double epsilon =  0.1;// -1.5
	double gamma   =  1.0;
	double q0      =  1.0;

// Parameters, variables and arrays

	int i, i_local, j, k, index, rank, size;

	std::vector <double> Vqy(Ny), Vqz(Nz);
	double scale = 1.0/NG;
	
	const int Nw    = 24;  // number of pts per wavelength

	const double Lx = Nx*2.0*M_PI/(q0*Nw); // Length X
	const double dx = Lx/Nx;
	const double Ly = Ny*2.0*M_PI/(q0*Nw); // Length Y
	const double dy = Ly/Ny;
	const double Lz = Nz*2.0*M_PI/(q0*Nw); // Length Z
	const double dz = Lz/Nz;

	double dt = 0.1;
	double dtd2 = dt/2.0;	

	double L1, sumA, sumA_local, sumB, sumB_local;

	const std::complex<double> iMag(0,1);

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
	std::vector<double> psiNew(size*2*alloc_local);
	std::vector<double> psiNew_local (2*alloc_local);
	std::vector<double> Nr_local (2*alloc_local);

	std::vector< std::complex<double> > psiq_local (alloc_local);
	std::vector< std::complex<double> > psiqNew_local (alloc_local);
	std::vector< std::complex<double> > Nq_local (alloc_local);
	std::vector< std::complex<double> > NqPast_local (alloc_local);

	std::vector< std::complex<double> > aLin (alloc_local);
	std::vector< std::complex<double> > C1 (alloc_local);
	std::vector< std::complex<double> > C2 (alloc_local);

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
			aLin[index] = epsilon - pow((pow(Vqx[i],2) + pow(Vqy[j],2)
						+ pow(Vqz[k],2) - q02),2);
			
			C1[index] = 1.0+dtd2*aLin[index];
			C2[index] = 1.0-dtd2*aLin[index];
		}}
	}


// Create forward FFTW plan

	planPsi = fftw_mpi_plan_dft_r2c_3d(Nx,Ny,Nz,
	            	    psi_local.data(),
				        reinterpret_cast<fftw_complex*>(psiq_local.data()),
				        MPI::COMM_WORLD,FFTW_PATIENT);

	planN = fftw_mpi_plan_dft_r2c_3d(Nx,Ny,Nz,
	            	    Nr_local.data(),
				        reinterpret_cast<fftw_complex*>(Nq_local.data()),
				        MPI::COMM_WORLD,FFTW_PATIENT);

// Create backward FFTW plan

	iPlanPsi = fftw_mpi_plan_dft_c2r_3d(Nx,Ny,Nz,
	             		reinterpret_cast<fftw_complex*>(psiqNew_local.data()),
				        psiNew_local.data(),
				        MPI::COMM_WORLD,FFTW_PATIENT);

// Inform allocated size for each rank

	std::cout << "Rank "<< rank << std::endl 
	<< "alloc_local: " << alloc_local <<  std::endl
	<< "local_n0: " << local_n0 << std::endl
	<< "local_0_start: " << local_0_start << std::endl;

std::ofstream growth;

if (rank == 0)
{
	std::ofstream growth("growth.dat");
}

for ( double mqZ = 0.80; mqZ < 2.01; mqZ = mqZ + 0.02)
{

// Initial condition

	for ( i_local = 0; i_local < local_n0; i_local++ ) 
	{
		i = i_local + local_0_start;

		for ( j = 0; j < Ny; j++ ) {
		for ( k = 0; k < Nz; k++ ) 
		{	
			index = (i_local*Ny + j) * (2*(Nz/2+1)) + k;
			psi_local[index] = 0.0001*cos(mqZ*k*dz);
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

	int time = 0;

	L1 = 1.0;

//	while ( L1 > 0.005) 
	for ( time = 0; time < 3; time++)
	{
//		MPI::COMM_WORLD.Barrier();

	//	time = time + 1;

		for ( i_local = 0; i_local < local_n0; i_local++ ) {
		for ( j = 0; j < Ny; j++ ) {
		for ( k = 0; k < Nz; k++ )
		{
			index = (i_local*Ny + j) * (2*(Nz/2+1)) + k;
			Nr_local[index] = beta*pow(psi_local[index],3) 
							- gamma*pow(psi_local[index],5);
		}}}

		fftw_execute(planPsi);

		fftw_execute(planN);

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

		NqPast_local = Nq_local;

		if (time == 2 )
		{
			MPI::COMM_WORLD.Gather(psi_local.data(),2*alloc_local,MPI::DOUBLE,
	         		       psi.data(),2*alloc_local, MPI::DOUBLE,0);

			MPI::COMM_WORLD.Gather(psiNew_local.data(),2*alloc_local,MPI::DOUBLE,
	         		       psiNew.data(),2*alloc_local, MPI::DOUBLE,0);

			if ( rank == 0 )
			{
				growth.open("growth.dat",std::ios_base::app); 
				assert(growth.is_open());

				i = Nx/2; j = Ny/2; k = Nz/2;

				index = (i*Ny + j) * (2*(Nz/2+1)) + k;
		
				double growthRate = (psiNew[index]-psi[index])/(dt*psi[index]);

				growth << growthRate << "\n ";

				growth.close();
			}
		}

		psi_local = psiNew_local;
	}

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
	fftw_destroy_plan(planN);
	fftw_destroy_plan(iPlanPsi);

	MPI::Finalize();

}
