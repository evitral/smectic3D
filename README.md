# smectic3D
FFT Phase Field Model for a 3D Smectic

1. pf3d.cpp : basic FFTW for the 3d smectic

2. pf3dST.cpp : Same as (1), but used before surfTrack (no real difference)

3. pf3dMorph.cpp : Same as (1), with hyperboloid like IC

4. surfTrack.cpp : Tracks surface velocity

5. cosNoAdvMorph.cpp : DCT for (3) IC, without advection

6. energyPsi.m : computes energy integral

7. pf3dVal.cpp : validation via wavenumber
