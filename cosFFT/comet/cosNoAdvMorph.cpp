/********************************************
 *                                          *
 *            cosNoAdvMorph.cpp             *
 *                                          *
 *     3D Phase Field + FFTW in parallel    *
 *     cos: DCT (Discrete FT)               *
 *     NoAdv: Advection is off              *
 *     Morph: Hyperboloid initial cond      *
 *                                          *
 *     Last mod: 08/18/2017                 *
 *     Author: Eduardo Vitral               *
 *                                          *
 ********************************************/

/* General */

#include <vector>
#include <cassert>
#include <cstdlib>       // std::exit()
#include <fftw3-mpi.h>

/* Input, output, string */

#include <string>
#include <iostream>
#include <iomanip>       // std::setw
#include <fstream>       // read, write

/* Math */

#include <algorithm>     // for max/min
#include <cmath>
#include <complex>

/* Time control (need c++11 for chrono) */

#include <ctime>
#include <cstdio>
#include <chrono>


/************** Compilation *****************

MSI:

module load intel ompi/intel

mpicxx -I /soft/fftw/intel-ompi/3.3-double/include -o code code.cpp
-L /soft/fftw/intel-ompi/3.3-double/lib -lfftw3_mpi -lfftw3 -lm -std=c++11 

COMET:

module load gnutools
module load intel/2016.3.210 mvapich2_ib 

mpicxx -I /opt/fftw/3.3.4/intel/mvapich2_ib/include -O2 -o code code.cpp 
-L /opt/fftw/3.3.4/intel/mvapich2_ib/lib -lfftw3_mpi -lfftw3 -lm -std=c++11 

********************************************/

/********************************************
 *                                          *
 *               FUNCTIONS                  *
 *                                          *
 *******************************************/    

double wallPotential(int z)
{
	double wP;
	double sigma = 0.0; // 1
	double z0 = 0.0001; //0.0001;
	wP = sigma*exp(-z/z0);
	return wP;
}


/********************************************
 *                                          *
 *                 MAIN                     *
 *                                          *
 *******************************************/    

int main(void) {

/* FFTW plans */

	fftw_plan planPsi, iPlanPsi, planN;

/* Indices and mpi related numbers */

	int i, j, k, index, t, i_local, rank, size;

/* Fourier space doubles */

	double mq2, opSH, aLin;

/* L1 related doubles + output */

	double L1, limL1, sumA, sumA_local, sumB, sumB_local;

	std::ofstream L1_output;

/* Load/save parameters */

	int load = 1;  // (load YES == 1)

	int swtPsi = 0;  // (switch: psi.dat/psiB.dat)

	std::string strPsi = "psi.dat";

	std::ofstream psiMid_output;
	
/* ptrdiff_t: integer type, optimizes large transforms 64bit machines */

	const ptrdiff_t Nx = 512, Ny = 512, Nz = 512;
	const ptrdiff_t NG = Nx*Ny*Nz;

	ptrdiff_t alloc_local, local_n0, local_0_start;

/* Constants and variables for morphologies (Nx = Ny = Nz) */

	const double mid = Nz/2; 
	const double aE = 270; // 80 // 312 // 432 // 248
	const double bE = 270; // 86 // 376 // 520 // 248

	double xs, ys, zs, ds;

/* Phase Field parameters */

	const double gamma =  1.0;
	const double beta  =  2.0;
	const double alpha =  1.0;
	const double ep    = -0.8; // -0.7 CHANGED !!!!!!!!!!!!!!
	const double q0    =  1.0;
	const double q02   = q0*q0;

/* Points per wavelength, time step */
	
	const int    Nw = 16;
	const double dt = 0.0005; // 0.0005 (nw 16)	
	const double dtd2  = dt/2;

/* System size and scaling for FFT */

	const double Lx = Nx*2.0*M_PI/(q0*Nw);
	const double dx = Lx/(Nx);
	const double Ly = Ny*2.0*M_PI/(q0*Nw);
	const double dy = Ly/(Ny);
	const double Lz = Nz*2.0*M_PI/(q0*Nw);
	const double dz = Lz/(Nz);

//	double scale = 0.125/((Nx-1)*(Ny-1)*(Nz-1));
	double scale = 0.125/((Nx)*(Ny)*(Nz));


/********************************************
 *                                          *
 *           Initialize MPI                 *
 *                                          *
 *******************************************/    

	MPI::Init();
	fftw_mpi_init();

	rank = MPI::COMM_WORLD.Get_rank();
	size = MPI::COMM_WORLD.Get_size();

	alloc_local = fftw_mpi_local_size_3d(Nx,Ny,Nz,MPI::COMM_WORLD,
			                		     &local_n0, &local_0_start);

/* Check: np should divide evenly into Nx, Ny and Nz */

/*

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

*/

/* Number of processors and initial time */

	if ( rank == 0 ) 
	{
		std::cout << "Using " << size << " processors." << std::endl;
	
    	time_t now = time(0);
    	char* dNow = ctime(&now);
    		   
    	std::cout << "The initial date and time is: " << dNow << std::endl;
	}

/********************************************
 *                                          *
 *              Containers                  *
 *                                          *
 *******************************************/    

/* Local data containers */

	std::vector <double> Vqx(local_n0), Vqy(Ny), Vqz(Nz);

	std::vector<double> C1(alloc_local);
	std::vector<double> C2(alloc_local);
	std::vector<double> psi_local(alloc_local);
	std::vector<double> psiq_local(alloc_local);
	std::vector<double> psiNew_local(alloc_local);
	std::vector<double> Nr_local(alloc_local);
	std::vector<double> Nq_local(alloc_local);
	std::vector<double> NqPast_local(alloc_local);

/* Local data containers (wall potential) */
	
	std::vector<double> wall(alloc_local);
	std::vector<double> substrate(alloc_local);
	
/* Global data containers */

	std::vector<double> psi(size*alloc_local);


/********************************************
 *                                          *
 *         Wavenumbers for r2r DCT          *
 *                                          *
 *   Note: for some reason the other one    *
 *   also seems to work, but I think this   *
 *   is the right definition.               *
 *                                          *
 *******************************************/


/* Wavenumbers (regular DFT) */

/*
	Vqx[0] = 0.0; Vqx[Nx/2] = 0.5*M_PI/dx;

	for ( i = 1; i < Nx/2; i++ )
	{
		Vqx[i] = 1.0*M_PI*i/(dx*Nx);
		Vqx[Nx/2+i] = -(Nx/2-i)*1.0*M_PI/(dx*Nx);
	}

	Vqy[0] = 0.0; Vqy[Ny/2] = 0.5*M_PI/dy;

	for ( j = 1; j < Ny/2; j++ )
	{
		Vqy[j] = 1.0*M_PI*j/(dy*Ny);
		Vqy[Ny/2+j] = -(Ny/2-j)*1.0*M_PI/(dy*Ny);
	}
*/


/* Wavenumbers (DCT) */

	for ( i_local = 0; i_local < local_n0; i_local++ ) 
	{	
		i = i_local + local_0_start;

	 	Vqx[i_local] = 1.0*M_PI*i/(dx*Nx);
	}	

	for ( j = 0; j < Ny; j++ )
	{
		Vqy[j] = M_PI*(j)/(dy*Ny);
	}

	for ( k = 0; k < Nz; k++ )
	{
		Vqz[k] = M_PI*(k)/(dz*Nz);
	}


/********************************************
 *                                          *
 *               FFTW plans                 *
 *                                          *
 *	 Notes:                           *
 *                                          *
 *   a. REDFT10 has REDFT01 as inverse      *
 *   + 2*N for scaling (in each dim).       *
 *   It seems to be the fastest one.        *
 *                                          *
 *   b. REDTF00 inverse is also REDTF00     *
 *   + 2*(N-1) for scaling (in each dim).   *
 *                                          *
 *******************************************/

	planPsi = fftw_mpi_plan_r2r_3d(Nx,Ny,Nz,
		  	  psi_local.data(),psiq_local.data(),MPI::COMM_WORLD,
		  	  FFTW_REDFT10,FFTW_REDFT10,FFTW_REDFT10,
		  	  FFTW_MEASURE);

	iPlanPsi = fftw_mpi_plan_r2r_3d(Nx,Ny,Nz,
		   	   psiq_local.data(),psiNew_local.data(),MPI::COMM_WORLD,
		   	   FFTW_REDFT01,FFTW_REDFT01,FFTW_REDFT01,
		       FFTW_MEASURE);

	planN = fftw_mpi_plan_r2r_3d(Nx,Ny,Nz,
			Nr_local.data(),Nq_local.data(),MPI::COMM_WORLD,
			FFTW_REDFT10,FFTW_REDFT10,FFTW_REDFT10,
	        FFTW_MEASURE);

/********************************************
 *                                          *
 *       Initial condition (New/Load)       *
 *                                          *
 *******************************************/

/* A. Initial condition - New */

	if ( load != 1 )
	{

	double Amp = 1.328; 

/*************** Not in use ****************

	double Qi  = 2.0; // Perturbation wavelength

	if ( (k > Nx/5) && ( k < 4*Nx/5))
	psi_local[index] = Amp*cos(q0*k*dz);
					 + Amp*0.5*sin(q0*k*dz)*(cos(Qi*i*dx)+cos(Qi*j*dy)); 
					 + Amp*0.5*(cos(Qi*i*dx)+cos(Qi*j*dy));
	
********************************************/


	for ( i_local = 0; i_local < local_n0; i_local++ ) 
	{
		i = i_local + local_0_start;

		for ( j = 0; j < Ny; j++ ) {
		for ( k = 0; k < Nz; k++ ) 
		{	
			index = (i_local*Ny + j) * Nz + k;
			if ( k <  bE + 1 ) // 18 110 // 24 232  // 62 450
			{		
				xs = i - mid;
				ys = j - mid;
				// zs = k + mid*3/4; 
				zs = k;
				// zs = k-mid for hyperboloid in the middle
				// zs = k for hyperboloid in the botton
				ds = sqrt(xs*xs+ys*ys);
				if (ds < mid)
				{
					if (sqrt(pow((ds-mid)/aE,2)+pow(zs/bE,2)) > 1)
					{
						psi_local[index] = 0.0;
					}
					else
					{
						psi_local[index] = Amp*cos(q0*dz*
						sqrt(pow((bE/aE)*(ds-mid),2)+zs*zs));
					}
				}
				else
				{
					if (abs(zs) < bE)
					{
						psi_local[index] = Amp*cos(q0*zs*dz);
					}
					else
					{
						psi_local[index] = 0.0;
					}
				}		 
			}
			else
			{
				psi_local[index] = 0.0;
			}
		}}
	} // close IC assign

/* Transfer IC data to global container psi in Rank 0 */

	MPI::COMM_WORLD.Barrier();

	MPI::COMM_WORLD.Gather(psi_local.data(),alloc_local,MPI::DOUBLE,
	         		       psi.data(),alloc_local, MPI::DOUBLE,0);

/* Output IC to file and create L1 output */

	if (rank == 0 )
	{	
		
		std::ofstream psiIC_output("psiIC.dat");
		assert(psiIC_output.is_open());
  
		for ( i = 0; i < Nx; i++ ) {
		for ( j = 0; j < Ny; j++ ) {
		for ( k = 0; k < Nz; k++ ) 
		{
			index = (i*Ny + j) * Nz + k;	
			psiIC_output << psi[index] << "\n ";
		}}}

		psiIC_output.close();		

		std::ofstream L1_output("L1.dat");
		assert(L1_output.is_open());
		L1_output.close();

		/** Create psiMid output **/

		std::ofstream psiMid_output("psiMid.dat");
		assert(psiMid_output.is_open());
		psiMid_output.close();

	}

	} // End: new psi (A)

	
/* B. Initial condition - Read profile data */

	if ( load == 1 )
	{

	if ( rank == 0 )
	{

	/** Open file and obtain IC for global psi **/
 
    	std::ifstream psidata("psiMove.dat");
    	assert(psidata.is_open());

    	std::cout << "Reading from the file" << std::endl;

		for ( i = 0; i < Nx; i++ ) {
		for ( j = 0; j < Ny; j++ ) {
		for ( k = 0; k < Nz; k++ ) 
		{
			index = (i*Ny + j) * Nz + k;			

			psidata >> psi[index];

		}}}

    	psidata.close();

	/** Create L1 output **/

		std::ofstream L1_output("L1.dat");
		assert(L1_output.is_open());
		L1_output.close();

	/** Create psiMid output **/

		std::ofstream psiMid_output("psiMid.dat");
		assert(psiMid_output.is_open());
		psiMid_output.close();
 
	}

	/** Scatter global psi data **/

	MPI::COMM_WORLD.Barrier();

	MPI::COMM_WORLD.Scatter(psi.data(),alloc_local,MPI::DOUBLE,
	         		       psi_local.data(),alloc_local, MPI::DOUBLE,0);

	} // End: load psi (B)

/********************************************
 *                                          *
 *         FS constants + 1st Nr            *
 *                                          *
 *   C1,C2: pointwise multiplication        *
 *          constants for Fourier Space     *
 *          LinOp (CrankNic/AdamsBash)      *
 *                                          *
 *   Nr_local: nonlinear terms (pre loop)   *
 *                                          *
 *******************************************/
 	
	// Generate psiq for substrate gradient penalty
	fftw_execute(planPsi);
	//
	
	for ( i_local = 0; i_local < local_n0; i_local++ ){
		
	i = i_local + local_0_start;

	for ( j = 0; j < Ny; j++ ) {
	for ( k = 0; k < Nz; k++ )
	{
		index =  (i_local*Ny + j)*Nz + k;
		mq2 = pow(Vqx[i_local],2)+pow(Vqy[j],2)+pow(Vqz[k],2);
		opSH = alpha*pow(mq2-q02,2);
		aLin = ep - opSH; // + (pow(Vqx[i_local],2)+pow(Vqy[j],2))*wallPotential(k);
		C1[index] = (1.0+dtd2*aLin);
		C2[index] = (1.0-dtd2*aLin);

		// Substrate penalty
		wall[index] = wallPotential(k);
		substrate[index] = -scale*(pow(Vqx[i_local],2)+pow(Vqy[j],2));
		psiq_local[index] = substrate[index]*psiq_local[index];
		//
		
		//		Nr_local[index] = beta*pow(psi_local[index],3)
		//  - gamma*pow(psi_local[index],5); // + wall[index];
	}}}

	// psiNew is the gradient penalty in real space
	fftw_execute(iPlanPsi);
	//

	// Compute Nr adding the substrate penalty * wall potential
	for ( i_local = 0; i_local < local_n0; i_local++ ){
		
	i = i_local + local_0_start;

	for ( j = 0; j < Ny; j++ ) {
	for ( k = 0; k < Nz; k++ )
	{
		index =  (i_local*Ny + j)*Nz + k;
		
		Nr_local[index] = beta*pow(psi_local[index],3)
		  - gamma*pow(psi_local[index],5) + psiNew_local[index]*wall[index]; // + wall[index];
	}}}
	//

	
/* Move Nr_local to Fourier Space */

	fftw_execute(planN);

/********************************************
 *                                          *
 *            Pre loop routine              *
 *                                          *
 *******************************************/

/* Pre loop values, indices */
	
//	sleep(5);

	L1 = 1.0;

	int countL1 = 0;

	int countSave = 0;

	limL1 = pow(10.0,-4);

	int nLoop = 0;

	MPI::COMM_WORLD.Barrier();

/* Pre loop announcement */

	std::clock_t startcputime; 
	auto wcts = std::chrono::system_clock::now();

	if ( rank == 0 )
	{
		time_t now = time(0);
    	char* dNow = ctime(&now);	   
    	std::cout << "The pre loop local date and time is: " 
		<< dNow << std::endl;
		startcputime = std::clock();	
	}


/********************************************
 *                                          *
 *   Time Loop (L1 as dynamics criterion)   *
 *                                          *
 *******************************************/

//	while ( countL1 < 0 )
	while (L1 > limL1)
	{

		countL1++;

/* Previous Nq_local is now NqPast_local  */

		NqPast_local = Nq_local;

/* Computing the substrate penalty */

		fftw_execute(planPsi);

		Nq_local = psiq_local;
		
		for ( i_local = 0; i_local < local_n0; i_local++ ){

		i = i_local + local_0_start;

		for ( j = 0; j < Ny; j++ ) {
		for ( k = 0; k < Nz; k++ ) 
		{
			index =  (i_local*Ny + j)*Nz + k;
			psiq_local[index] = substrate[index]*psiq_local[index];
		}}}

		fftw_execute(iPlanPsi);

		psiq_local = Nq_local;
		
/* Compute current Nr_local */

		for ( i_local = 0; i_local < local_n0; i_local++ ){

		i = i_local + local_0_start;

		for ( j = 0; j < Ny; j++ ) {
		for ( k = 0; k < Nz; k++ ) 
		{
			index =  (i_local*Ny + j)*Nz + k;
			Nr_local[index] = beta*pow(psi_local[index],3)
			  - gamma*pow(psi_local[index],5) + psiNew_local[index]*wall[index]; // + wall[index];			  
		}}}

/* Obtain current Nq_local */

		fftw_execute(planN);	

/* Moves current psi to Fourier Space */
// was already computed in the substrate penalty section
//		fftw_execute(planPsi);

/* Compute new psi in Fourier Space (CN/AB scheme) */

		for ( i_local = 0; i_local < local_n0; i_local++ ){
		
		i = i_local + local_0_start;

		for ( j = 0; j < Ny; j++ ) {
		for ( k = 0; k < Nz; k++ )
		{
			index =  (i_local*Ny + j)*Nz + k;
			psiq_local[index] = scale*(C1[index]*psiq_local[index]
			+ dtd2*(3.0*Nq_local[index]-NqPast_local[index]))/C2[index];
		}}}	

/* Obtain new psi in real space */

		fftw_execute(iPlanPsi);

/* Compute L1 (under count condition) */

		if ( countL1 == 100 )
		{
    		 
			sumA_local = 0.0; sumB_local = 0.0;
			sumA = 0.0;       sumB = 0.0;

			for ( i_local = 0; i_local < local_n0; i_local++ ) {
			for ( j = 0; j < Ny; j++ ) {
			for ( k = 0; k < Nz; k++ )
			{
				index = (i_local*Ny + j) * Nz + k;
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

/* Save psi (under count condition) */

			if ( countSave == 10 )
			{
				MPI::COMM_WORLD.Gather(psiNew_local.data(),alloc_local,
				MPI::DOUBLE,psi.data(),alloc_local, MPI::DOUBLE,0);

				if (rank == 0 )
				{
			
					psiMid_output.open("psiMid.dat",std::ios_base::app);
     			    assert(psiMid_output.is_open());

					j = 256;
					
					for ( i = 0; i < Nx; i++ ) {
					for ( k = 0; k < Nz; k++ ) 
					{
						index = (i*Ny + j) * Nz + k;
						psiMid_output << psi[index] << "\n";
					}}

					psiMid_output.close();
			

				  
	/** Switch between two save files **/

					if ( swtPsi == 0) {
						strPsi = "psi.dat";
						swtPsi = 1;
					}
					else {
						strPsi = "psiB.dat";
						swtPsi = 0;						
					}

					std::ofstream psi_output(strPsi.c_str());
					assert(psi_output.is_open());
	
					for ( i = 0; i < Nx; i++ ) {
					for ( j = 0; j < Ny; j++ ) {
					for ( k = 0; k < Nz; k++ ) 
					{
						index = (i*Ny + j) * Nz + k;
						psi_output << psi[index] << "\n ";
					}}}

					psi_output.close();

	/** Inform date and time after each save psi **/

	    		    time_t now = time(0);
	    		    char* dNow = ctime(&now);
					nLoop++;    		   

	    		    std::cout << "The loop " << 1000*nLoop 
					<< " local date and time is: " << dNow << std::endl;

				}

				countSave = 0;

			} // End: countSave block
		
		} // End: countL1 block

		psi_local = psiNew_local;

	} // End: time loop

/********************************************
 *                                          *
 *         Post Time Loop routine           *
 *                                          *
 *******************************************/

	if ( rank == 0 )
	{
		time_t now = time(0);
    	char* dNow = ctime(&now);	   
    	std::cout << "The post loop local date and time is: " 
		<< dNow << std::endl;
		double cpu_duration = (std::clock() - startcputime) / (double)CLOCKS_PER_SEC;
		std::cout << "Finished in " << cpu_duration << " seconds [CPU Clock] " << std::endl;
		std::chrono::duration<double> wctduration = (std::chrono::system_clock::now() - wcts);
		std::cout << "Finished in " << wctduration.count() << " seconds [Wall Clock] " << std::endl;
	}

/* Gather and save psi */

	MPI::COMM_WORLD.Barrier();

	MPI::COMM_WORLD.Gather(psi_local.data(),alloc_local,MPI::DOUBLE,
	         		       psi.data(),alloc_local, MPI::DOUBLE,0);

	if ( rank == 0) {

/*

	std::ofstream psi_output("psi.dat");
	assert(psi_output.is_open());

	for ( i = 0; i < Nx; i++ ) {
	for ( j = 0; j < Ny; j++ ) {
	for ( k = 0; k < Nz; k++ )
	{
		index = (i*Nx + j)*Nz + k;
		psi_output << psi[index] << "\n ";
	}}}

	psi_output.close();

*/

	std::cout << "Max: " << *std::max_element( psi.begin(),psi.end()) <<
	std::endl;

	std::cout << "Min: " << *std::min_element( psi.begin(),psi.end()) <<
	std::endl;

	std::cout << "Done!" << std::endl;
	}

/* Destroy FFTW plans, cleanup */

  	fftw_destroy_plan(planPsi);
  	fftw_destroy_plan(iPlanPsi);
  	fftw_destroy_plan(planN);

  	fftw_cleanup();


	
/* Finalize MPI */

	MPI::Finalize();

} // END
