#include <iostream>
#include <string>
#include <cmath>
#include <fstream>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <chrono>

const double R = 0.4;


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


int main(int argc, char* argv[]) {

   int numEvents;
   //Default value
   if (argc < 2) numEvents = 1000;
   else numEvents = std::stod(argv[1]);

   
   const int cols = 2101;
   const size_t totElements = (size_t)numEvents * cols;
   
   std::vector<float> data(totElements);
   
   //Read binary file
   std::ifstream inFile("../data.bin", std::ios::binary);
   if(!inFile){
      std::cerr << "Error opening the file " << std::endl;
      return 1; 
   }

   inFile.read(reinterpret_cast<char*>(data.data()), totElements * sizeof(float));
   inFile.close();




   //Save information regarding the entire collision time
   std::ofstream Collision("Collision.csv");
   if (!Collision.is_open()) {
      std::cerr << "Error: Could not open CSV file for writing!" << std::endl;
      return 1;
   }
   Collision << "TotalTime,DistanceTime,MinimumTime,JetIdentificationTime\n";


   //Loop of collisions
   for(int collision = 0; collision < numEvents; ++collision){

      std::cout << collision << std::endl;

      //Retrieve the correct row 
      float *ptr = &data[collision * cols];

      //Jets and Active particles vector
      std::vector<Particle> activeParticles;
      std::vector<Particle> Jet;


      //Vector holding timing informations: Benchmark purposes
      double CollisionEvent = 0;
      double Distance = 0;
      double Minimum = 0;
      double JetIdentification = 0;


      //store in activeParticles each particle (pT, eta, phi)
      for(int i = 0; i + 2 < cols; i += 3){
         if(ptr[i] == 0) break;

         activeParticles.push_back({(double)ptr[i], (double)ptr[i+1], (double)ptr[i+2]});
      }

      auto Istart = std::chrono::high_resolution_clock::now();


      while (activeParticles.size() != 0){
         
         int numParticles = activeParticles.size();

         //Vector of structure to hold geometric information on the position
         std::vector<DistanceTag> distance_ij;
         std::vector<DistanceTag> distance_iB;

         //----------------------------COMPUTING DISTANCES------------------------

         auto start = std::chrono::high_resolution_clock::now();

         for(int i = 0; i < numParticles; ++i){

            //Compute the distance between particle i and beam: distance, position i, -1
            distance_iB.push_back({D_iB(i, activeParticles), i, -1});

            for(int j = i+1; j < numParticles; ++j){
               
               //Compute distance between particle i and j and fill vector
               distance_ij.push_back({D_ij(i, j, activeParticles), i, j});

            }
         }

         auto end = std::chrono::high_resolution_clock::now();
         std::chrono::duration<double> elapsed = end - start;

         Distance += elapsed.count();


         //----------------------------FINDING THE MINIMUM------------------------
         start = std::chrono::high_resolution_clock::now();

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

         end = std::chrono::high_resolution_clock::now();
         elapsed = end - start;

         Minimum += elapsed.count();



         //----------------------------LOOKING FOR THE JET------------------------
         start = std::chrono::high_resolution_clock::now();


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

            double pT_i = activeParticles[pIdx_i].pT, pT_j = activeParticles[pIdx_j].pT;
            double eta_i = activeParticles[pIdx_i].eta, eta_j = activeParticles[pIdx_j].eta;
            double phi_i = activeParticles[pIdx_i].phi, phi_j = activeParticles[pIdx_j].phi;

            int first = std::min(pIdx_i, pIdx_j);
            int second = std::max(pIdx_i, pIdx_j);

            //add the four momenta
            double pT = pT_i + pT_j;
            double eta = (pT_i * eta_i + pT_j * eta_j)/pT;
            double phi = (pT_i * phi_i + pT_j * phi_j)/pT;


            activeParticles.erase(activeParticles.begin() + second);
            activeParticles.erase(activeParticles.begin() + first);

            Particle newParticle = {pT, eta, phi};
            activeParticles.push_back(newParticle);


         }

         end = std::chrono::high_resolution_clock::now();
         elapsed = end - start;

         JetIdentification += elapsed.count();

      }

         auto Iend = std::chrono::high_resolution_clock::now();
         std::chrono::duration<double> elapsed = Iend - Istart;

         CollisionEvent = elapsed.count();

         Collision << CollisionEvent << "," << Distance << "," << Minimum << "," << JetIdentification << "\n";

   }
   
   
   return 0;
}


   double D_ij(int i, int j, const std::vector<Particle>& activeParticles){

      //Defining variables
      double p_i = activeParticles[i].pT, p_j = activeParticles[j].pT;
      double eta_i = activeParticles[i].eta, eta_j = activeParticles[j].eta;
      double phi_i = activeParticles[i].phi, phi_j = activeParticles[j].phi;
      
      //Compute deltaR squared 
      double DeltaR2 = pow(eta_i - eta_j, 2) + pow(phi_i - phi_j, 2);
      
      double d_ij = std::min(pow(p_i, -2), pow(p_j, -2))* DeltaR2 /pow(R, 2);
      
      return d_ij;
   }

   double D_iB(int i, const std::vector<Particle>& activeParticles){

      double p_i = activeParticles[i].pT;
      double d_iB = pow(p_i, -2);

      return d_iB;
   }




