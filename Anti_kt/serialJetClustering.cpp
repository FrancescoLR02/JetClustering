#include <iostream>
#include <string>
#include <cmath>
#include <fstream>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <chrono>

const double R = 0.4;
constexpr double PI = 3.14159265358979323846;


//Structure of partice 
struct Particle{
   double pT, eta, phi;
};

//Struct for the distance (keeps track of the index of the particle)
struct DistanceTag{
   double dist;
   int i, j;
};

//Functions declaration
double D_ij(int i, int j, const std::vector<Particle>& activeParticle);
double D_iB(int i, const std::vector<Particle>& activeParticle);
Particle newParticle(const Particle &p1, const Particle &p2);



int main(int argc, char* argv[]) {

   int numEvents;
   int nColl;
   int totalEvents = 100000;

   //Default value
   if (argc < 3) {
      numEvents = 1000;
      nColl = -1;
   }
   else {
      numEvents = std::stod(argv[1]);
      nColl = std::stod(argv[2]);
   }


   const int cols = 2101;
   const size_t totElements = (size_t)totalEvents * cols;

   std::vector<float> data(totElements);

   //load the binary file
   std::ifstream inFile("../data.bin", std::ios::binary);
   if(!inFile){
      std::cerr << "Error opening the file " << std::endl;
      return 1; 
   }

   inFile.read(reinterpret_cast<char*>(data.data()), totElements * sizeof(float));
   inFile.close();

   //outer loop: collisions
   for(int collision = 0; collision < numEvents; ++collision){

      //std::cout << collision << std::endl;

      //Retrieve the correct row 
      float *ptr = &data[collision * cols];

      //Jets and Active particles vector
      std::vector<Particle> activeParticles;
      std::vector<Particle> Jet;


      //store in activeParticles each particle (pT, eta, phi)
      for(int i = 0; i + 2 < cols; i += 3){
         if(ptr[i] == 0) break;

         activeParticles.push_back({(double)ptr[i], (double)ptr[i+1], (double)ptr[i+2]});
      }

      //Verifying the complexity of the algorithm
      if(nColl != -1){
         if(activeParticles.size() > nColl){
            activeParticles.resize(nColl);
         }
         else continue;
      }

      //Iterative steps
      while (activeParticles.size() != 0){
         
         //Euclidean Distance calculation
         int numParticles = activeParticles.size();

         //Vector of structure to hold geometric information on the position
         std::vector<DistanceTag> distance_ij;
         std::vector<DistanceTag> distance_iB;

         for(int i = 0; i < numParticles; ++i){

            //Compute the distance between particle i and beam: distance, position i, -1
            distance_iB.push_back({D_iB(i, activeParticles), i, -1});

            for(int j = i+1; j < numParticles; ++j){
               
               //Compute distance between particle i and j and fill vector
               distance_ij.push_back({D_ij(i, j, activeParticles), i, j});

            }
         }

         //minimum element for both vector for particle-particle and particle-Beam
         //Defining a new operator to find the minimum value in the struct
         auto pB = std::min_element(distance_iB.begin(), distance_iB.end(), 
               [](const DistanceTag& a, const DistanceTag& b) {return a.dist < b.dist;});
   
         bool isJet = false;
         auto pp = distance_ij.begin();
               
         //if the vector of distances is empty only 1 particle remains
         if (distance_ij.empty()){
            isJet = true;
         }
         else{
            pp = std::min_element(distance_ij.begin(), distance_ij.end(), 
                  [](const DistanceTag& a, const DistanceTag& b) {return a.dist < b.dist;});

            if(pB->dist < pp->dist){
               isJet = true;
            }
         }

         //if isJet is true, add to Jet structure
         if(isJet){

            int pIdx = pB->i;

            //Add the particle to the jet vector
            Jet.push_back(activeParticles[pIdx]);

            //erasing the particle from the column vector
            activeParticles.erase(activeParticles.begin() + pIdx);

         }

         //else: not a jet, merging the two four-momenta
         else{

            int pIdx_i = pp->i;
            int pIdx_j = pp->j;

            Particle merged = newParticle(activeParticles[pIdx_i], activeParticles[pIdx_j]);

            int first = std::min(pIdx_i, pIdx_j);
            int second = std::max(pIdx_i, pIdx_j);


            activeParticles.erase(activeParticles.begin() + second);
            activeParticles.erase(activeParticles.begin() + first);

            activeParticles.push_back(merged);
         }

      }
   }
   
   
   return 0;
}


   double D_ij(int i, int j, const std::vector<Particle>& activeParticles){

      //Defining variables
      double p_i = activeParticles[i].pT, p_j = activeParticles[j].pT;
      double eta_i = activeParticles[i].eta, eta_j = activeParticles[j].eta;
      double phi_i = activeParticles[i].phi, phi_j = activeParticles[j].phi;

      //account for the periodicity of phi: wrapped Delta phi: result in [-pi, pi]
      double dphi = std::remainder(phi_i - phi_j, 2.0 * PI);
      
      //Compute deltaR squared 
      double DeltaR2 = pow(eta_i - eta_j, 2) + pow(dphi, 2);
      
      double d_ij = std::min(pow(p_i, -2), pow(p_j, -2))* DeltaR2 /pow(R, 2);
      
      return d_ij;
   }

   double D_iB(int i, const std::vector<Particle>& activeParticles){

      double p_i = activeParticles[i].pT;
      double d_iB = pow(p_i, -2);

      return d_iB;
   }

   Particle newParticle(const Particle &p1, const Particle &p2){

      double tot_pT = p1.pT + p2.pT;

      double eta = (p1.pT * p1.eta + p2.pT * p2.eta) / tot_pT;
      
      //account for periodicity of phi
      double dphi = std::remainder(p2.phi - p1.phi, 2.0 * PI);
      double phi = p1.phi + (p2.pT / tot_pT) * dphi;

      phi = std::remainder(phi, 2.0 * PI);

      return {tot_pT, eta, phi};
   }




