#include <iostream>
#include <string>
#include <cmath>
#include <fstream>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <chrono>

const double R = 0.4;
const int cols = 2101;


//Structure of partice 
struct Particle{
   int id;
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


int main() {

   //Read binary file
   const int numEvents = 1000;
   const size_t totElements = (size_t)numEvents * cols;

   std::vector<float> data(totElements);

   //Preallocation of the memory for the data
   std::vector<std::vector<Particle>> allResults(numEvents);

   std::ifstream inFile("data.bin", std::ios::binary);
   if(!inFile){
      std::cerr << "Error opening the file " << std::endl;
      return 1; 
   }

   inFile.read(reinterpret_cast<char*>(data.data()), totElements * sizeof(float));
   inFile.close();


   //Loop of collisions
   #pragma omp parallel for
   for(int collision = 0; collision < numEvents; ++collision){

      //Retrieve the correct row 
      float *ptr = &data[collision * cols];

      //Jets and Active particles vector
      std::vector<Particle> activeParticles;
      std::vector<Particle> Jet;


      //store in activeParticles each particle (pT, eta, phi)
      for(int i = 0; i + 2 < cols; i += 3){
         if(ptr[i] == 0) break;

         activeParticles.push_back({(int)collision, (double)ptr[i], (double)ptr[i+1], (double)ptr[i+2]});
      }


      
      while (activeParticles.size() != 0){
         
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

            Particle newParticle = {collision, pT, eta, phi};
            activeParticles.push_back(newParticle);


         }

      }

      allResults[collision] = Jet;

      //std::cout << "Event " << collision << " finished! Found " << Jet.size() << " Jets." << std::endl;

   }

   //Save jets in memory
   std::ofstream output("ReconstructedJetDir.csv");
   if (!output.is_open()) {
      std::cerr << "Error: Could not open CSV file for writing!" << std::endl;
      return 1;
   }
   output << "EventID,pT,Eta,Phi\n";

   //Write the vectors in memory
   for(int i = 0; i < numEvents; ++i){
      for (const auto& j : allResults[i]) {
         output << j.id << "," << j.pT << "," << j.eta << "," << j.phi << "\n";
      }
   }
   output.close();
   
   
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




