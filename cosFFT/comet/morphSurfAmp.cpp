/********************************************
 *                                          *
 *            morphSurfAmp.cpp              *
 *                                          *
 *     3D Phase Field + FFTW in parallel    *
 *     Surf: tracks smectic surface         *
 *     Amp: obtain amplitude of the PF      *
 *     (cos): DCT (Discrete FT) for PF      *
 *     (sin): DST for BLM                   *
 *     (Adv): Advection is on               *
 *     Morph: Toroidal initial cond         *
 *                                          *
 *     Note: this code obtains all          *
 *     surface info from the amplitude      *
 *     of the phase field, by taking its    *
 *     average in space                     *
 *                                          *
 *     Last mod: 01/22/2018                 *
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

double trapz (int nstart, int nend, double h,std::vector<double> psiInt)
{
	double trapz = 0;
	int nmax = nend-nstart;
	for (int n = 0; n < nmax ; n++ )
	{
		trapz = trapz + psiInt[n] + psiInt[n+1];
	}
	trapz = 0.5*h*trapz;
	return trapz;
}



/********************************************
 *                                          *
 *                 MAIN                     *
 *                                          *
 *******************************************/    

int main(void) {

/* FFTW plans */

	fftw_plan planPsi, iPlanPsi, planN, iPlanPsiSH, 
		      planSx, planSy, planSz, iPlanSx, iPlanSy, iPlanSz,
			  iPlanPsiDxx, iPlanPsiDyy, iPlanPsiDxy, 
			  iPlanPsiDzz, iPlanPsiDxz, iPlanPsiDyz, iPlanDTpsi;

/* Indices and mpi related numbers */

	int i, j, k, index, index0, index1, index2, index3, t, i_local, rank, size;

/* Fourier space doubles */

	double mq2, dotSqVq;

/* Surf Track and Psi Average related */

	int track, nlambda, istart, iend, jstart, jend, kstart, kend, i2, j2, k2;

	double gradVal, lambda3, psiRef, l1, l2, l3;

/* Load/save parameters */

	int load = 1;  // (load YES == 1)

	int swtPsi = 0;  // (switch: psi.dat/psiB.dat)

	std::string strPsi = "psi.dat";

/* ptrdiff_t: integer type, optimizes large transforms 64bit machines */

	const ptrdiff_t Nx = 512, Ny = 512, Nz = 512;
	const ptrdiff_t NG = Nx*Ny*Nz;

	ptrdiff_t alloc_local, local_n0, local_0_start;

/* Constants and variables for morphologies (Nx = Ny = Nz) */

	const double mid = Nz/2; 
	const double aE = 432; // 80 // 312
	const double bE = 520; // 86 // 376
	
	double xs, ys, zs, ds;

/* Phase Field parameters */

	const double gamma =  1.0;
	const double beta  =  2.0;
	const double alpha =  1.0;
	const double ep    = -0.7;
	const double q0    =  1.0;
	const double q02   = q0*q0;

/* Balance of Linear Momentum parameters */

	double eta = 1.0;
	double rho = 1.0;

/* Points per wavelength, time step */
	
	const int    Nw = 16;
	const double dt = 0.0005; // 0.0005	
	const double dtd2  = dt/2;

/* System size and scaling for FFT */
/* Note: the logical system size os 2x this one 
 * when using DCT and DST */

	const double Lx = Nx*2.0*M_PI/(q0*Nw);
	const double dx = Lx/(Nx);
	const double Ly = Ny*2.0*M_PI/(q0*Nw);
	const double dy = Ly/(Ny);
	const double Lz = Nz*2.0*M_PI/(q0*Nw);
	const double dz = Lz/(Nz);

	const double tdx = 2*dx;
	const double tdy = 2*dy;
	const double tdz = 2*dz;

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

	double alloc_surf = local_n0*Ny;

/* Check: np should divide evenly into Nx, Ny and Nz */

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

	std::vector <double> Vsx(local_n0), Vsy(Ny), Vsz(Nz);

	std::vector<double> C1(alloc_local);
	std::vector<double> C2(alloc_local);
	std::vector<double> psi_local(alloc_local);
	std::vector<double> psiq_local(alloc_local);
	std::vector<double> psiNew_local(alloc_local);
	std::vector<double> Nv_local(alloc_local);
	std::vector<double> NqPast_local(alloc_local);
	std::vector<double> psiGradx_local(alloc_local);
	std::vector<double> psiGrady_local(alloc_local);
	std::vector<double> psiGradz_local(alloc_local);
	std::vector<double> opSH(alloc_local);
	std::vector<double> psiSH_local(alloc_local);
	std::vector<double> Sx_local(alloc_local);
	std::vector<double> Sy_local(alloc_local);
	std::vector<double> Sz_local(alloc_local);
	std::vector<double> CM1(alloc_local);
	std::vector<double> CM2x(alloc_local);
	std::vector<double> CM2y(alloc_local);
	std::vector<double> CM2z(alloc_local);

	// Global

	std::vector<double> psiGradx(size*alloc_local);
	std::vector<double> psi(size*alloc_local);

/* Surf Track additions */
	
	std::vector<double> aLin(alloc_local);
	std::vector<double> dTpsi_local(alloc_local);
	std::vector<double> psiDxx_local(alloc_local);
	std::vector<double> psiDyy_local(alloc_local);
	std::vector<double> psiDxy_local(alloc_local);

	std::vector<double> psiDzz_local(alloc_local);
	std::vector<double> psiDxz_local(alloc_local);
	std::vector<double> psiDyz_local(alloc_local);

	std::vector<double> surfZ_local(alloc_surf);
	std::vector<double> velSurf_local(alloc_surf);
	std::vector<double> curvH_local(alloc_surf);
	std::vector<double> curvK_local(alloc_surf);

	std::vector<double> surfZ(size*alloc_surf);
	std::vector<double> velSurf(size*alloc_surf);
	std::vector<double> curvH(size*alloc_surf);
	std::vector<double> curvK(size*alloc_surf);

	std::vector<double> normGradPsi_local(alloc_local);
	std::vector<double> normGradx_local(alloc_local);
	std::vector<double> normGrady_local(alloc_local);
	std::vector<double> normGradz_local(alloc_local);

	// Global

	std::vector<double> normGradPsi(size*alloc_local);
	std::vector<double> normGradx(size*alloc_local);

/* Psi Average related */

	std::vector<double> psiAbsX(Nw+1);
	std::vector<double> psiAbsY(Nw+1);
	std::vector<double> psiAbsZ(Nw+1);

	// Global
	
	std::vector<double> psiAvg(size*alloc_local);


/********************************************
 *                                          *
 *       Wavenumbers for r2r DCT/DST        *
 *                                          *
 *******************************************/


/* Wavenumbers (regular DFT) */

/*
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

	Vqy[0] = 0.0; Vqy[Ny/2] = 1.0*M_PI/dy;

	for ( j = 1; j < Ny/2; j++ ) 
	{
    	Vqy[j] = 2.0*M_PI*j/(dy*Ny);
    	Vqy[Ny/2+j] = -(Ny/2-j)*2.0*M_PI/(dy*Ny);
	}

	Vqz[0] = 0.0; Vqz[Nz/2] = 1.0*M_PI/dz;

	for ( k = 1; k < Nz/2; k++ ) 
	{
		Vqz[k] = 2.0*M_PI*k/(dz*Nz);
		Vqz[Nz/2+k] = -(Nz/2-k)*2.0*M_PI/(dz*Nz);
	}
*/

/* Wavenumbers (DCT) */

	for ( i_local = 0; i_local < local_n0; i_local++ ) 
	{	
		i = i_local + local_0_start;

	 	Vqx[i_local] = 1.0*M_PI*(i)/(dx*Nx); // (dx*Nx)
	}	

	for ( j = 0; j < Ny; j++ )
	{
		Vqy[j] = M_PI*(j)/(dy*Ny);
	}

	for ( k = 0; k < Nz; k++ )
	{
		Vqz[k] = M_PI*(k)/(dz*Nz);
	}

/* Wavenumbers (DST) */

	for ( i_local = 0; i_local < local_n0; i_local++ ) 
	{	
		i = i_local + local_0_start;

	 	Vsx[i_local] = 1.0*M_PI*(i+1)/(dx*Nx); // (dx*Nx)
	}	

	for ( j = 0; j < Ny; j++ )
	{
		Vsy[j] = M_PI*(j+1)/(dy*Ny);
	}

	for ( k = 0; k < Nz; k++ )
	{
		Vsz[k] = M_PI*(k+1)/(dz*Nz);
	}

/********************************************
 *                                          *
 *               FFTW plans                 *
 *                                          *
 *	 Notes:                                 *
 *                                          *
 *   a. R*DFT10 has R*DFT01 as inverse      *
 *   + 2*N for scaling (in each dim).       *
 *   It seems to be the fastest one.        *
 *                                          *
 *   b. R*DTF00 inverse is also R*DTF00     *
 *   + 2*(N-1) for scaling (in each dim).   *
 *                                          *
 *   c. In-place r2r transforms have        *
 *   absolutely no further complications    *
 *                                          *
 *   (*) E for DCT, O for DST               *
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
			Nv_local.data(),Nv_local.data(),MPI::COMM_WORLD,
			FFTW_REDFT10,FFTW_REDFT10,FFTW_REDFT10,
	        FFTW_MEASURE);

	iPlanPsiSH = fftw_mpi_plan_r2r_3d(Nx,Ny,Nz,
		   	     psiSH_local.data(),psiSH_local.data(),MPI::COMM_WORLD,
		   	     FFTW_REDFT01,FFTW_REDFT01,FFTW_REDFT01,
		         FFTW_MEASURE);

	planSx = fftw_mpi_plan_r2r_3d(Nx,Ny,Nz,
		     Sx_local.data(),Sx_local.data(),MPI::COMM_WORLD,
			 FFTW_RODFT10,FFTW_RODFT10,FFTW_RODFT10,
	         FFTW_MEASURE);

	planSy = fftw_mpi_plan_r2r_3d(Nx,Ny,Nz,
			 Sy_local.data(),Sy_local.data(),MPI::COMM_WORLD,
			 FFTW_RODFT10,FFTW_RODFT10,FFTW_RODFT10,
	         FFTW_MEASURE);

	planSz = fftw_mpi_plan_r2r_3d(Nx,Ny,Nz,
			 Sz_local.data(),Sz_local.data(),MPI::COMM_WORLD,
			 FFTW_RODFT10,FFTW_RODFT10,FFTW_RODFT10,
	         FFTW_MEASURE);

	iPlanSx = fftw_mpi_plan_r2r_3d(Nx,Ny,Nz,
		   	  Sx_local.data(),Sx_local.data(),MPI::COMM_WORLD,
		   	  FFTW_RODFT01,FFTW_RODFT01,FFTW_RODFT01,
		      FFTW_MEASURE);

	iPlanSy = fftw_mpi_plan_r2r_3d(Nx,Ny,Nz,
		   	  Sy_local.data(),Sy_local.data(),MPI::COMM_WORLD,
		   	  FFTW_RODFT01,FFTW_RODFT01,FFTW_RODFT01,
		      FFTW_MEASURE);

	iPlanSz = fftw_mpi_plan_r2r_3d(Nx,Ny,Nz,
		   	  Sz_local.data(),Sz_local.data(),MPI::COMM_WORLD,
		   	  FFTW_RODFT01,FFTW_RODFT01,FFTW_RODFT01,
		      FFTW_MEASURE);

	iPlanPsiDxx = fftw_mpi_plan_r2r_3d(Nx,Ny,Nz,
		   	      psiDxx_local.data(),psiDxx_local.data(),MPI::COMM_WORLD,
		   	      FFTW_REDFT01,FFTW_REDFT01,FFTW_REDFT01,
		          FFTW_MEASURE);

	iPlanPsiDyy = fftw_mpi_plan_r2r_3d(Nx,Ny,Nz,
		   	      psiDyy_local.data(),psiDyy_local.data(),MPI::COMM_WORLD,
		   	      FFTW_REDFT01,FFTW_REDFT01,FFTW_REDFT01,
		          FFTW_MEASURE);

	iPlanPsiDxy = fftw_mpi_plan_r2r_3d(Nx,Ny,Nz,
		   	      psiDxy_local.data(),psiDxy_local.data(),MPI::COMM_WORLD,
		   	      FFTW_REDFT01,FFTW_REDFT01,FFTW_REDFT01,
		          FFTW_MEASURE);

	iPlanPsiDzz = fftw_mpi_plan_r2r_3d(Nx,Ny,Nz,
		   	      psiDzz_local.data(),psiDzz_local.data(),MPI::COMM_WORLD,
		   	      FFTW_REDFT01,FFTW_REDFT01,FFTW_REDFT01,
		          FFTW_MEASURE);

	iPlanPsiDxz = fftw_mpi_plan_r2r_3d(Nx,Ny,Nz,
		   	      psiDxz_local.data(),psiDxz_local.data(),MPI::COMM_WORLD,
		   	      FFTW_REDFT01,FFTW_REDFT01,FFTW_REDFT01,
		          FFTW_MEASURE);

	iPlanPsiDyz = fftw_mpi_plan_r2r_3d(Nx,Ny,Nz,
		   	      psiDyz_local.data(),psiDyz_local.data(),MPI::COMM_WORLD,
		   	      FFTW_REDFT01,FFTW_REDFT01,FFTW_REDFT01,
		          FFTW_MEASURE);

	iPlanDTpsi = fftw_mpi_plan_r2r_3d(Nx,Ny,Nz,
		   	     dTpsi_local.data(),dTpsi_local.data(),MPI::COMM_WORLD,
		   	     FFTW_REDFT01,FFTW_REDFT01,FFTW_REDFT01,
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
				zs = k + mid*3/4; 
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
	}

	} // End: new psi (A)


/* B. Initial condition - Read profile data */

	if ( load == 1 )
	{

	if ( rank == 0 )
	{

	/** Open file and obtain IC for global psi **/
 
    	std::ifstream psidata("/home/evitral/cosMorph/psiMove/psiMoveE0d8.dat");
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
 *   CM1,CM2: pointwise multiplication      *
 *            constants for Fourier Space   *
 *            velocity computation          *
 *                                          *
 *   Nr_local: nonlinear terms (pre loop)   *
 *                                          *
 *******************************************/


	for ( i_local = 0; i_local < local_n0; i_local++ ){
		
	i = i_local + local_0_start;

	for ( j = 0; j < Ny; j++ ) {
	for ( k = 0; k < Nz; k++ )
	{
		index =  (i_local*Ny + j)*Nz + k;
		mq2 = pow(Vqx[i_local],2)+pow(Vqy[j],2)+pow(Vqz[k],2);
		opSH[index] = alpha*pow(mq2-q02,2);
		aLin[index] = ep - opSH[index];
		opSH[index] = scale*opSH[index];
		C1[index] = (1.0+dtd2*aLin[index]);
		C2[index] = scale/(1.0-dtd2*aLin[index]);

		mq2 = pow(Vsx[i_local],2)+pow(Vsy[j],2)+pow(Vsz[k],2);
		CM1[index] = scale/(rho*eta*mq2);
		CM2x[index] = Vsx[i_local]/mq2; 
		CM2y[index] = Vsy[j]/mq2;
		CM2z[index] = Vsz[k]/mq2;

		Nv_local[index] = beta*pow(psi_local[index],3)
			            - gamma*pow(psi_local[index],5);		
	}}}

/* Move Nr_local to Fourier Space */

	fftw_execute(planN);

/********************************************
 *                                          *
 *            Pre loop routine              *
 *                                          *
 *******************************************/

/* Pre loop values, indices */

	int countQuit = 0;

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

	while ( countQuit < 1 )
//	while (L1 > limL1)
	{

		countQuit++;

/* Previous Nq_local is now NqPast_local  */

		NqPast_local = Nv_local;

/* Moves current psi to Fourier Space */

		fftw_execute(planPsi);

/* psiSH */

		for ( i_local = 0; i_local < local_n0; i_local++ ){
		for ( j = 0; j < Ny; j++ ) {
		for ( k = 0; k < Nz; k++ )
		{
			index =  (i_local*Ny + j)*Nz + k;
			psiSH_local[index] = opSH[index]*psiq_local[index];
		}}}

		fftw_execute(iPlanPsiSH);

/* Compute gradient of psi */

		if (rank == 0){
	
			for ( i = 1; i < Nx-1; i++ ){
			for ( j = 0; j < Ny; j++ ){
			for ( k = 0; k < Nz; k++ ) 
			{
				index =  (i*Ny + j)*Nz + k;
				index1 = ((i-1)*Ny + j)*Nz + k;
				index2 = ((i+1)*Ny + j)*Nz + k;
				psiGradx[index] = (psi[index2]-psi[index1])/tdx;
			}}}

			i = 0;
			for ( j = 0; j < Ny; j++ ){
			for ( k = 0; k < Nz; k++ ) 
			{
				index =  (i*Ny + j)*Nz + k;
				index2 = ((i+1)*Ny + j)*Nz + k;
				psiGradx[index] = 2*(psi[index2]-psi[index])/tdx;
			}}

			i = Nx-1;
			for ( j = 0; j < Ny; j++ ){
			for ( k = 0; k < Nz; k++ ) 
			{
				index =  (i*Ny + j)*Nz + k;
				index1 = ((i-1)*Ny + j)*Nz + k;
				psiGradx[index] = 2*(psi[index]-psi[index1])/tdx;
			}}

		}

	/** Scatter psiGradx data **/

		MPI::COMM_WORLD.Barrier();

		MPI::COMM_WORLD.Scatter(psiGradx.data(),alloc_local,MPI::DOUBLE,
	    	     		       psiGradx_local.data(),alloc_local, MPI::DOUBLE,0);

/* Compute psiGrady;z and Sx;y;z and move to FS */


		for ( i_local = 0; i_local < local_n0; i_local++ ){
		for ( j = 0; j < Ny; j++ ) {
		for ( k = 0; k < Nz; k++ ) 
		{
			index =  (i_local*Ny + j)*Nz + k;
			index1 = (i_local*Ny + (j-1))*Nz + k;
			index2 = (i_local*Ny + (j+1))*Nz + k;
			if ((j!=0) && (j!=Ny-1)){ 
				psiGrady_local[index] = (psi_local[index2]-psi_local[index1])/tdy;
			} else if (j==0){ 
				psiGrady_local[index] = 2*(psi_local[index2]-psi_local[index])/tdy;
			} else {
			  	psiGrady_local[index] = 2*(psi_local[index]-psi_local[index1])/tdy;
			}
			if ((k!=0) && (k!=Nz-1)){
				psiGradz_local[index] = (psi_local[index+1]-psi_local[index-1])/tdz;
			} else if(k==0){ 
				psiGradz_local[index] = 2*(psi_local[index+1]-psi_local[index])/tdz; 
			} else {
				psiGradz_local[index] = 2*(psi_local[index]-psi_local[index-1])/tdz;
			}
			Sx_local[index] = psiSH_local[index]*psiGradx_local[index];
			Sy_local[index] = psiSH_local[index]*psiGrady_local[index];
			Sz_local[index] = psiSH_local[index]*psiGradz_local[index];
		}}}
		
		fftw_execute(planSx);
		fftw_execute(planSy);
		fftw_execute(planSz);

/* Generate a matrix containing the 2-norm of grad psi */

		for ( i_local = 0; i_local < local_n0; i_local++ ){
		for ( j = 0; j < Ny; j++ ) {
		for ( k = 0; k < Nz; k++ ) 
		{
			index = (i_local*Ny + j)*Nz + k;
			normGradPsi_local[index] = sqrt(psiGradx_local[index]*psiGradx_local[index]
					   + psiGrady_local[index]*psiGrady_local[index]
					   + psiGradz_local[index]*psiGradz_local[index]);

		}}}

	/** Gather normGradPsi data **/

		MPI::COMM_WORLD.Barrier();

		MPI::COMM_WORLD.Gather(normGradPsi_local.data(),alloc_local,
			MPI::DOUBLE,normGradPsi.data(),alloc_local, MPI::DOUBLE,0);

/* Compute gradient of Norm Grad Psi */

		if (rank == 0){
	
			for ( i = 1; i < Nx-1; i++ ){
			for ( j = 0; j < Ny; j++ ){
			for ( k = 0; k < Nz; k++ ) 
			{
				index =  (i*Ny + j)*Nz + k;
				index1 = ((i-1)*Ny + j)*Nz + k;
				index2 = ((i+1)*Ny + j)*Nz + k;
				normGradx[index] = (normGradPsi[index2]-normGradPsi[index1])/tdx;
			}}}

			i = 0;
			for ( j = 0; j < Ny; j++ ){
			for ( k = 0; k < Nz; k++ ) 
			{
				index =  (i*Ny + j)*Nz + k;
				index2 = ((i+1)*Ny + j)*Nz + k;
				normGradx[index] = 2*(normGradPsi[index2]-normGradPsi[index])/tdx;
			}}

			i = Nx-1;
			for ( j = 0; j < Ny; j++ ){
			for ( k = 0; k < Nz; k++ ) 
			{
				index =  (i*Ny + j)*Nz + k;
				index1 = ((i-1)*Ny + j)*Nz + k;
				normGradx[index] = 2*(normGradPsi[index]-normGradPsi[index1])/tdx;
			}}

		}

	/** Scatter normGradx data **/

		MPI::COMM_WORLD.Barrier();

		MPI::COMM_WORLD.Scatter(normGradx.data(),alloc_local,MPI::DOUBLE,
	    	     		       normGradx_local.data(),alloc_local, MPI::DOUBLE,0);

/* Compute psiGrady;z and Sx;y;z and move to FS */


		for ( i_local = 0; i_local < local_n0; i_local++ ){
		for ( j = 0; j < Ny; j++ ) {
		for ( k = 0; k < Nz; k++ ) 
		{
			index =  (i_local*Ny + j)*Nz + k;
			index1 = (i_local*Ny + (j-1))*Nz + k;
			index2 = (i_local*Ny + (j+1))*Nz + k;
			if ((j!=0) && (j!=Ny-1)){ 
				normGrady_local[index] = (normGradPsi_local[index2]-normGradPsi_local[index1])/tdy;
			} else if (j==0){ 
				normGrady_local[index] = 2*(normGradPsi_local[index2]-normGradPsi_local[index])/tdy;
			} else {
			  	normGrady_local[index] = 2*(normGradPsi_local[index]-normGradPsi_local[index1])/tdy;
			}
			if ((k!=0) && (k!=Nz-1)){
				normGradz_local[index] = (normGradPsi_local[index+1]-normGradPsi_local[index-1])/tdz;
			} else if(k==0){ 
				normGradz_local[index] = 2*(normGradPsi_local[index+1]-normGradPsi_local[index])/tdz; 
			} else {
				normGradz_local[index] = 2*(normGradPsi_local[index]-normGradPsi_local[index-1])/tdz;
			}
		}}}
	

/* Compute velocity in Fourier Space and transform back */

		for ( i_local = 0; i_local < local_n0; i_local++ ) {
		for ( j = 0; j < Ny; j++ ) {
		for ( k = 0; k < Nz; k++ )
		{
			index = (i_local*Ny + j) * Nz + k;
			dotSqVq = Vsx[i_local]*Sx_local[index]
				    + Vsy[j]*Sy_local[index]
				    + Vsz[k]*Sz_local[index]; 
			Sx_local[index] = CM1[index]*(Sx_local[index]
							 - CM2x[index]*dotSqVq);
			Sy_local[index] = CM1[index]*(Sy_local[index]
							 - CM2y[index]*dotSqVq);
			Sz_local[index] = CM1[index]*(Sz_local[index]
							 - CM2z[index]*dotSqVq);
		}}}

		fftw_execute(iPlanSx);
		fftw_execute(iPlanSy);
		fftw_execute(iPlanSz);

/* Compute current Nr_local */

		for ( i_local = 0; i_local < local_n0; i_local++ ){

		i = i_local + local_0_start;

		for ( j = 0; j < Ny; j++ ) {
		for ( k = 0; k < Nz; k++ ) 
		{
			index =  (i_local*Ny + j)*Nz + k;
			Nv_local[index] = beta*pow(psi_local[index],3)
			  	            - gamma*pow(psi_local[index],5);
							//- Sx_local[index]*psiGradx_local[index]
							//- Sy_local[index]*psiGrady_local[index]
							//- Sz_local[index]*psiGradz_local[index];
			}}}

/* Obtain current Nq_local */

		fftw_execute(planN);	

/* Compute new psi in Fourier Space (CN/AB scheme) */

		for ( i_local = 0; i_local < local_n0; i_local++ ){
		
		i = i_local + local_0_start;

		for ( j = 0; j < Ny; j++ ) {
		for ( k = 0; k < Nz; k++ )
		{
			index =  (i_local*Ny + j)*Nz + k;

			dTpsi_local[index] = scale*(aLin[index]*psiq_local[index]+Nv_local[index]);
			
			psiDxx_local[index] = -scale*Vqx[i_local]*Vqx[i_local]*psiq_local[index];
		
			psiDyy_local[index] = -scale*Vqy[j]*Vqy[j]*psiq_local[index];
		
			psiDzz_local[index] = -scale*Vqz[k]*Vqz[k]*psiq_local[index];

			psiDxy_local[index] = -scale*Vqx[i_local]*Vqy[j]*psiq_local[index];
	
			psiDxz_local[index] = -scale*Vqx[i_local]*Vqz[k]*psiq_local[index];

			psiDyz_local[index] = -scale*Vqy[j]*Vqz[k]*psiq_local[index];

			psiq_local[index] = C2[index]*(C1[index]*psiq_local[index]
			+ dtd2*(3.0*Nv_local[index]-NqPast_local[index])); // C2 = scale/oldC2

		}}}	

/* Obtain new psi in real space */

		fftw_execute(iPlanPsi);
		fftw_execute(iPlanPsiDxx);
		fftw_execute(iPlanPsiDyy);
		fftw_execute(iPlanPsiDzz);
	
		fftw_execute(iPlanPsiDxy);
		fftw_execute(iPlanPsiDxz);
		fftw_execute(iPlanPsiDyz);
	
		fftw_execute(iPlanDTpsi);

		MPI::COMM_WORLD.Gather(psiNew_local.data(),alloc_local,
			MPI::DOUBLE,psi.data(),alloc_local, MPI::DOUBLE,0);

/* Surf track: since I computed evth for present psi, we use it */

//		psi_local = psiNew_local;

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

/* Obtain the psi average */

	if( rank == 0 )
	{

	// First, compute reference psi

		nlambda = Nw/2;

		istart = nlambda;
		jstart = nlambda;
		kstart = nlambda;
		iend   = 3*nlambda;
		jend   = 3*nlambda;
		kend   = 3*nlambda;

		index3 = 0;

		for (k2 = kstart; k2 < kend + 1; k2++)
		{

		index2 = 0;

		for (j2 = jstart; j2 < jend + 1; j2++)
		{

		index1 = 0;

		for (i2 = istart; i2 < iend + 1; i2++)
		{
			index =  (i2*Ny + j2)*Nz + k2;
			psiAbsX[index1] = psi[index]*psi[index];
			index1++;
		}

		psiAbsY[index2] = trapz(istart,iend,dx,psiAbsX);
		
		index2++;

		}
		
		psiAbsZ[index3] = trapz(jstart,jend,dy,psiAbsY);

		index3++;

		}

		psiRef = trapz(kstart,kend,dz,psiAbsZ);

	// Now, compute psi average matrix

		lambda3 = 8.0*pow(M_PI,3);   // for q0 = 1

		for (k = 0; k < Nz; k++)
		{

		kstart = k - nlambda;
		kend = k + nlambda;
		if (kstart < 0) { kstart = 0;}
		if (kend > Nz-1) { kend = Nz-1;}

		for (j = 0; j < Ny; j++)
		{
	
		jstart = j - nlambda;
		jend = j + nlambda;
		if (jstart < 0) { jstart = 0;}
		if (jend > Ny-1) { jend = Ny-1;}

		for (i = 0; i < Nx; i++)
		{
		
		istart = i - nlambda;
		iend = i + nlambda;
		if (istart < 0) { istart = 0;}
		if (iend > Nx-1) { iend = Nx-1;}

			index0 =  (i*Ny + j)*Nz + k;

			l1 = (iend-istart)*dx;
			l2 = (jend-jstart)*dy;
			l3 = (kend-kstart)*dz;

			index3 = 0;

			for (k2 = kstart; k2 < kend + 1; k2++)
			{

			index2 = 0;

			for (j2 = jstart; j2 < jend + 1; j2++)
			{

			index1 = 0;

			for (i2 = istart; i2 < iend + 1; i2++)
			{
				index =  (i2*Ny + j2)*Nz + k2;
				psiAbsX[index1] = psi[index]*psi[index];
				index1++;
			}

			psiAbsY[index2] = trapz(istart,iend,dx,psiAbsX);
		
			index2++;

			}
		
			psiAbsZ[index3] = trapz(jstart,jend,dy,psiAbsY);

			index3++;

			}

			psiAvg[index0] = trapz(kstart,kend,dz,psiAbsZ)/(psiRef*l1*l2*l3/lambda3);

		}}}

	std::ofstream psi_output("psiAvg.dat");
	assert(psi_output.is_open());

	for ( i = 0; i < Nx; i++ ) {
	for ( j = 0; j < Ny; j++ ) {
	for ( k = 0; k < Nz; k++ )
	{
		index = (i*Ny + j)*Nz + k;
		psi_output << psiAvg[index] << "\n ";
	}}}

	psi_output.close();

	}

	// Compute curvatures
	
	for ( i_local = 0; i_local < local_n0; i_local++ ) {

		for ( j = 0; j < Ny; j++ ) {
			
		track = 0;
		index2 = i_local*Ny + j;

		for ( k = Nz-1; k > -1; k-- ) 
		{

			index = (i_local*Ny + j) * Nz + k;

			if ( psi_local[index] > 0.7 & track == 0 ) // 0.7 ; results are better when looking for this 0
			{
				track = 1;
	    	}
			if ( psi_local[index] < 0.0 & track == 1 ) //std::abs(...) > 0.7
			{

				k2 = k;

				if( std::abs(psi_local[index]) > std::abs(psi_local[index+1]) ) // >
				{
					index = index + 1;
					k2 = k + 1;
				}
				surfZ_local[index2] = k2;
				
				gradVal = normGradPsi_local[index];
				
				velSurf_local[index2] = dTpsi_local[index]/gradVal;

				// Mixed second order derivates do not work with the DCT, so...
				
				if( j> 0 & j < (Ny-1) )
				{
				psiDxy_local[index] = (psiGradx_local[(i_local*Ny + j+1) * Nz + k2]-psiGradx_local[(i_local*Ny + j-1) * Nz + k2])/tdy;
				}			
				if(j==0)
				{
				psiDxy_local[index] = 2*(psiGradx_local[(i_local*Ny + 1) * Nz + k2]-psiGradx_local[(i_local*Ny) * Nz + k2])/tdy;
				}
				if(j==Ny-1)
				{
				psiDxy_local[index] = 2*(psiGradx_local[(i_local*Ny + j) * Nz + k2]-psiGradx_local[(i_local*Ny+j-1) * Nz + k2])/tdy;
				}

				if(k > 0 & k < (Nz-1) )
				{
				psiDxz_local[index] = (psiGradx_local[index+1]-psiGradx_local[index-1])/tdz;
				psiDyz_local[index] = (psiGrady_local[index+1]-psiGrady_local[index-1])/tdz;
				}			
				if(k == 0  )
				{
				psiDxz_local[index] = 2*(psiGradx_local[index+1]-psiGradx_local[index])/tdz;
				psiDyz_local[index] = 2*(psiGrady_local[index+1]-psiGrady_local[index])/tdz;
				}
				if(k == (Nz-1) )
				{
				psiDxz_local[index] = 2*(psiGradx_local[index]-psiGradx_local[index-1])/tdz;
				psiDyz_local[index] = 2*(psiGrady_local[index]-psiGrady_local[index-1])/tdz;
				}
				
				// Proper way to numerically compute H and K
				// (Megrabov 2014, On divergence representations ..)

				curvH_local[index2] =
					((pow(psiGrady_local[index],2)+pow(psiGradz_local[index],2))*psiDxx_local[index]
					 +(pow(psiGradx_local[index],2)+pow(psiGradz_local[index],2))*psiDyy_local[index]
					 +(pow(psiGradx_local[index],2)+pow(psiGrady_local[index],2))*psiDzz_local[index]
					 -2*(psiGradx_local[index]*psiGrady_local[index]*psiDxy_local[index]
						 +psiGradx_local[index]*psiGradz_local[index]*psiDxz_local[index]
						 +psiGrady_local[index]*psiGradz_local[index]*psiDyz_local[index]))
					/ pow(gradVal,3);
				
				curvK_local[index2] =
					(pow(psiGradz_local[index],2)*(psiDxx_local[index]*psiDyy_local[index]-pow(psiDxy_local[index],2))
					 +pow(psiGradx_local[index],2)*(psiDyy_local[index]*psiDzz_local[index]-pow(psiDyz_local[index],2))
					 +pow(psiGrady_local[index],2)*(psiDxx_local[index]*psiDzz_local[index]-pow(psiDxz_local[index],2))
					 +2*(psiGrady_local[index]*psiDxy_local[index]*(psiGradz_local[index]*psiDxz_local[index]-psiGradx_local[index]*psiDzz_local[index])
						 +psiGradx_local[index]*psiDxz_local[index]*(psiGrady_local[index]*psiDyz_local[index]-psiGradz_local[index]*psiDyy_local[index])
						 +psiGradz_local[index]*psiDyz_local[index]*(psiGradx_local[index]*psiDxy_local[index]-psiGrady_local[index]*psiDxx_local[index])
						 ))/pow(gradVal,4);

				track = 2;
			}
		}}
	}

	MPI::COMM_WORLD.Barrier();

	MPI::COMM_WORLD.Gather(psi_local.data(),alloc_local,MPI::DOUBLE,
						   psi.data(),alloc_local, MPI::DOUBLE,0);

	MPI::COMM_WORLD.Gather(surfZ_local.data(),alloc_surf,MPI::DOUBLE,
	         		       surfZ.data(),alloc_surf, MPI::DOUBLE,0);

	MPI::COMM_WORLD.Gather(velSurf_local.data(),alloc_surf,MPI::DOUBLE,
	         		       velSurf.data(),alloc_surf, MPI::DOUBLE,0);

	MPI::COMM_WORLD.Gather(curvH_local.data(),alloc_surf,MPI::DOUBLE,
	         		       curvH.data(),alloc_surf, MPI::DOUBLE,0);

	MPI::COMM_WORLD.Gather(curvK_local.data(),alloc_surf,MPI::DOUBLE,
	         		       curvK.data(),alloc_surf, MPI::DOUBLE,0);


	if ( rank == 0 )
	{
		std::ofstream surf_output("surfPsi.dat");
		std::ofstream velS_output("velSurf.dat");
		std::ofstream curvH_output("curvH.dat");
		std::ofstream curvK_output("curvK.dat");

		assert(surf_output.is_open());
		assert(velS_output.is_open());
		assert(curvH_output.is_open());
		assert(curvK_output.is_open());

		for ( i = 0; i < Nx; i++ ) {
		for ( j = 0; j < Ny; j++ ) {

			index = i*Ny + j;

			surf_output << surfZ[index] << "\n";
			velS_output << velSurf[index] << "\n ";
			curvH_output << curvH[index] << "\n ";
			curvK_output << curvK[index] << "\n ";

		}}

		surf_output.close();
		velS_output.close();
		curvH_output.close();
		curvK_output.close();	
	
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

	fftw_destroy_plan(iPlanPsiSH);

	fftw_destroy_plan(planSx);
	fftw_destroy_plan(planSy);
	fftw_destroy_plan(planSz);
	fftw_destroy_plan(iPlanSx);
	fftw_destroy_plan(iPlanSy);
	fftw_destroy_plan(iPlanSz);

	fftw_destroy_plan(iPlanPsiDxx);
	fftw_destroy_plan(iPlanPsiDyy);
	fftw_destroy_plan(iPlanPsiDxy);

	fftw_destroy_plan(iPlanPsiDzz);
	fftw_destroy_plan(iPlanPsiDxz);
	fftw_destroy_plan(iPlanPsiDyz);

	fftw_destroy_plan(iPlanDTpsi);

  	fftw_cleanup();

/* Finalize MPI */

	MPI::Finalize();

} // END
