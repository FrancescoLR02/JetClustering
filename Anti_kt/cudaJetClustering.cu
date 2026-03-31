#include <iostream>
#include <string>
#include <cmath>
#include <fstream>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <chrono>
#include <numeric>

#define THREADS_PER_BLOCK 128
#define BLOCKS 3


const double R = 0.4;
const float PI = 3.14159265f;
const float TWO_PI = 6.28318530f;



//Cuda kernel, on the device. The parameters are respectvly input and output vectors
__global__ void JetClusteringKernel(const float *pT, const float *eta, const float *phi, const int *nPar, 
                                    float *pT_J, float *eta_J, float *phi_J, int *nPar_J, int maxPar ){

   //Idx define the spefiic collision: eaco collision happens in a SM
   auto idx = blockIdx.x;

   //Global offset in the 1D vector 
   int globalIdx = idx * maxPar;

   
   //Define the number of active particles -> particles in not yet merged or jets
   __shared__ int activeParticles;

   if(threadIdx.x == 0){
      activeParticles = nPar[idx];
   }

   //Thread 0 wrtes, all the other need to read activeParticles
   __syncthreads();

   //fefine shared memory: from global memory to cache 
   __shared__ float s_pT[700];
   __shared__ float s_eta[700];
   __shared__ float s_phi[700];

   //Fill the shared memory
   for(int i = threadIdx.x; i < activeParticles; i += blockDim.x){
      s_pT[i] = pT[globalIdx + i];
      s_eta[i] = eta[globalIdx + i];
      s_phi[i] = phi[globalIdx + i];
   }

   //synch threads -> wait for every thread to copy in the shared memory
   __syncthreads();

   //Define arrya to compare the minimum distances to find the absolute minima
   __shared__ double s_minDist[128];
   __shared__ int s_minIdx_i[128];
   __shared__ int s_minIdx_j[128];

   
   //Define memory space for found jets
   int foundJets;
   if(threadIdx.x == 0) foundJets = 0;

   //GRID STRIDE LOOP
   //The idea is the every thread takes a subset of products of collisino, compute the distances and find the minimum in the subset
   //Then at the end all the threads will hold their minimum. The abolute minima is found applying parallel reduction
   while(activeParticles > 0){

      //Define the initial measure and the initial idx
      double minDist = 9e10;
      int minIdx_i = -1;
      int minIdx_j = -1;


      for(int i = threadIdx.x; i < activeParticles; i += blockDim.x){

         //Particle-Beam distance:
         double pT_i = s_pT[i];
         double d_iB = 1.0 / (pT_i * pT_i);

         //check whether this is the minimum distance in the grid 
         if(d_iB < minDist){
            minDist = d_iB;
            
            //store the index (i, j) of the particle with the minimum distance (j = -1 distance particle-beam)
            minIdx_i = i;
            minIdx_j = -1;
         }

         //Distance Particle-Particle
         for(int j = i + 1; j < activeParticles; ++j){
            
            double pT_j = s_pT[j];

            //minimum inverse squared momenta:
            double pT2_min = fmin(1.0 / (pT_i * pT_i), 1.0 / (pT_j * pT_j));

            double DeltaEta = s_eta[i] - s_eta[j];

            //periodic boundary condition for phi
            double DeltaPhi = s_phi[i] - s_phi[j];

            if (DeltaPhi > PI) {
               DeltaPhi -= TWO_PI;
            } 
            else if (DeltaPhi < -PI) {
               DeltaPhi += TWO_PI;
            }

            //Computing D_ij
            double DeltaR2 = (DeltaEta * DeltaEta) + (DeltaPhi * DeltaPhi);
            double d_ij = pT2_min * DeltaR2 /(R*R);

            if(d_ij < minDist){
               minDist = d_ij;
               minIdx_i = i;
               minIdx_j = j;
            }
         }
      }

      s_minDist[threadIdx.x] = minDist;
      s_minIdx_i[threadIdx.x] = minIdx_i;
      s_minIdx_j[threadIdx.x] = minIdx_j;

      //wait for all threads to compute their "local" minima
      __syncthreads();

      //Apply parallel reduction to find the sbolute minimum
      //128 -> 64 -> 32 -> ... -> 1
      for(int s = blockDim.x / 2; s > 0; s /= 2){
         if(threadIdx.x < s){

            //pair distance comparison
            if(s_minDist[threadIdx.x + s] < s_minDist[threadIdx.x]){
               
               s_minDist[threadIdx.x] = s_minDist[threadIdx.x + s];
               s_minIdx_i[threadIdx.x] = s_minIdx_i[threadIdx.x + s];
               s_minIdx_j[threadIdx.x] = s_minIdx_j[threadIdx.x + s];
            }

         }

         __syncthreads();
      }

      //Only thread 0 works here, otherwise race condition
      if(threadIdx.x == 0){

         int best_i = s_minIdx_i[0];
         int best_j = s_minIdx_j[0];

         //belong to Particle-Beam --> Jet
         if(best_j == -1){

            //write to global memory
            int outIdx = globalIdx + foundJets;

            pT_J[outIdx] = s_pT[best_i];
            eta_J[outIdx] = s_eta[best_i];
            phi_J[outIdx] = s_phi[best_i];

            foundJets++;

            //Swap the best_i at the edìnd of the array
            int last = activeParticles - 1;
            s_pT[best_i] = s_pT[last];
            s_eta[best_i] = s_eta[last];
            s_phi[best_i] = s_phi[last];


            //Update the number of active particles
            activeParticles--;
         }

         //Not a jet, need to merge four momenta
         else{

            double new_pT = s_pT[best_i] + s_pT[best_j];
            double new_eta = (s_pT[best_i]*s_eta[best_i] + s_pT[best_j]*s_eta[best_j]) / new_pT;
            
            //PBC for phi
            double dphi = s_phi[best_j] - s_phi[best_i];

            if (dphi > PI) {
               dphi -= TWO_PI;
            } 
            else if (dphi < -PI) {
               dphi += TWO_PI;
            }

            //compute the new phi
            double new_phi = s_phi[best_i] + (s_pT[best_j] / new_pT) * dphi;

            //Wrap the final averaged phi back to [-pi, pi]
            if (new_phi > PI) {
               new_phi -= TWO_PI;
            } 
            else if (new_phi <= -PI) {
               new_phi += TWO_PI;
            }

            //overwrite best_i with the new value
            s_pT[best_i] = new_pT;
            s_eta[best_i] = new_eta;
            s_phi[best_i] = new_phi;

            //swap best_j in last position
            int last = activeParticles - 1;
            s_pT[best_j] = s_pT[last];
            s_eta[best_j] = s_eta[last];
            s_phi[best_j] = s_phi[last]; 

            //Update acive particles
            activeParticles--;

         }

         nPar_J[idx] = foundJets;

      }

      __syncthreads();

   }
}



int main(int argc, char* argv[]) {

   int numCollision;
   //Default value
   if (argc < 2) numCollision = 100000;
   else numCollision = std::stod(argv[1]);


   const int cols = 2101;
   const int maxPar = cols / 3;
   const size_t totElements = (size_t)numCollision * cols;

   std::vector<float> data(totElements);

   std::ifstream inFile("data.bin", std::ios::binary);
   if(!inFile){
      std::cerr << "Error opening the file " << std::endl;
      return 1; 
   }

   inFile.read(reinterpret_cast<char*>(data.data()), totElements * sizeof(float));
   inFile.close();


   //Save jets in memory
   std::ofstream output("ReconstructedJet.csv");
   if (!output.is_open()) {
      std::cerr << "Error: Could not open CSV file for writing!" << std::endl;
      return 1;
   }
   output << "EventID,pT,Eta,Phi\n";



   //vectors of size Nx700, ensures contiguous memory allocation
   std::vector<float> pT(numCollision * maxPar);
   std::vector<float> eta(numCollision * maxPar);
   std::vector<float> phi(numCollision * maxPar);
   std::vector<int> nPar(numCollision);

   //initialize output vectors 
   std::vector<float> pT_Jet(numCollision * maxPar);
   std::vector<float> eta_Jet(numCollision * maxPar);
   std::vector<float> phi_Jet(numCollision * maxPar);
   std::vector<int> nPar_Jet(numCollision);


   //Loop of collisions
   for(int collision = 0; collision < numCollision; ++collision){

      //Retrieve the correct row 
      float *ptr = &data[collision * cols];
      int particleCount = 0;

      //store in activeParticles each particle (pT, eta, phi)
      for(int i = 0; i + 2 < cols; i += 3){
         if(ptr[i] == 0) break;

         //Track the correct index for the 1D vector 
         int idx = (collision * maxPar) + particleCount;

         pT[idx] = (float)ptr[i];
         eta[idx] = (float)ptr[i+1];
         phi[idx] = (float)ptr[i+2];

         particleCount++;
      }

      //Number of particle produced by each collision
      nPar[collision] = particleCount;

   }

   //computing size of vectors 
   size_t floatVec = numCollision * maxPar * sizeof(float);
   size_t intVec = numCollision * sizeof(int);

   //input pointers to device
   float *dev_pT, *dev_eta, *dev_phi;
   int *dev_nPar;

   //output ointers to device
   float *dev_pT_Jet, *dev_eta_Jet, *dev_phi_Jet;
   int *dev_nPar_Jet;

   //Memory allocation: input
   cudaMalloc((void **)&dev_pT, floatVec);
   cudaMalloc((void **)&dev_eta, floatVec);
   cudaMalloc((void **)&dev_phi, floatVec);
   cudaMalloc((void **)&dev_nPar, intVec);

   
   //Memory allocation: output
   cudaMalloc((void **)&dev_pT_Jet, floatVec);
   cudaMalloc((void **)&dev_eta_Jet, floatVec);
   cudaMalloc((void **)&dev_phi_Jet, floatVec);
   cudaMalloc((void **)&dev_nPar_Jet, intVec);

   
   //input vector from Host to Device
   cudaMemcpy(dev_pT, pT.data(), floatVec, cudaMemcpyHostToDevice);
   cudaMemcpy(dev_eta, eta.data(), floatVec, cudaMemcpyHostToDevice);
   cudaMemcpy(dev_phi, phi.data(), floatVec, cudaMemcpyHostToDevice);
   cudaMemcpy(dev_nPar, nPar.data(), intVec, cudaMemcpyHostToDevice);

   //determine how many threads and block to launch
   int nBlocks = numCollision;
   int nThreadsPerBlock = THREADS_PER_BLOCK;

   //Kernel
   JetClusteringKernel<<<nBlocks, nThreadsPerBlock>>>(dev_pT, dev_eta, dev_phi, dev_nPar, 
                                                      dev_pT_Jet, dev_eta_Jet, dev_phi_Jet, dev_nPar_Jet, maxPar);

   //Retrieve the data from Device to Host
   cudaMemcpy(pT_Jet.data(), dev_pT_Jet, floatVec, cudaMemcpyDeviceToHost);
   cudaMemcpy(eta_Jet.data(), dev_eta_Jet, floatVec, cudaMemcpyDeviceToHost);
   cudaMemcpy(phi_Jet.data(), dev_phi_Jet, floatVec, cudaMemcpyDeviceToHost);
   cudaMemcpy(nPar_Jet.data(), dev_nPar_Jet, intVec, cudaMemcpyDeviceToHost);

   cudaFree(dev_pT);
   cudaFree(dev_eta);
   cudaFree(dev_phi);
   cudaFree(dev_nPar);

   cudaFree(dev_pT_Jet);
   cudaFree(dev_eta_Jet);
   cudaFree(dev_phi_Jet);
   cudaFree(dev_nPar_Jet);


   for(int event = 0; event < numCollision; ++event){
      for(int i = 0; i < nPar_Jet[event]; ++i){

         //Takes the jets in the vector 
         int idx = (event * maxPar) + i;
         output << event << "," << pT_Jet[idx] << "," << eta_Jet[idx] << "," << phi_Jet[idx] << "\n";
      }
   }

   output.close();
   
   
   return 0;
}
