#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <algorithm>
#include <filesystem>
#include "H5Cpp.h"

using namespace H5;

const double R = 0.4;

const std::string FILE_NAME("U_events_anomalydetection_Z_XY_qqq.h5");
const std::string DATASET_NAME("df/block0_values");


//Structure of partice 
struct Particle{
   double pT, eta, phi;
};

//Struct for the distance
struct DistanceTag{
   double dist;
   int i, j;
};

//Functions declaration
double D_ij(int i, int j, const std::vector<Particle>& activeParticle);
double D_iB(int i, const std::vector<Particle>& activeParticle);


int main() {

   H5File file(FILE_NAME, H5F_ACC_RDONLY);
   DataSet dataset = file.openDataSet(DATASET_NAME);
   DataSpace dataspace = dataset.getSpace();

   hsize_t dims[2];
   dataspace.getSimpleExtentDims(dims, NULL);
   size_t rows = dims[0];
   size_t cols = dims[1];

   //Read each line of the dataset:
   hsize_t offset[2] = {0, 0};
   hsize_t count[2] = {1, cols};

   //define a single vector for a specific row
   std::vector<float> column(cols);

   //Define memory space
   DataSpace memspace(2, count);

   
   std::vector<Particle> Jet;

   std::ofstream output("ReconstructedJet.csv");
   if (!output.is_open()) {
      std::cerr << "Error: Could not open CSV file for writing!" << std::endl;
      return 1;
   }
   output << "pT,Eta,Phi\n";


   //Loop of collisions
   for(int collision = 0; collision < 20000; ++collision){

      //i-th row
      offset[0] = collision;

      dataspace.selectHyperslab(H5S_SELECT_SET, count, offset);
      dataset.read(column.data(), PredType::NATIVE_FLOAT, memspace, dataspace);

      //Active particles vector
      std::vector<Particle> activeParticles;

      
      for(int i = 0; i + 2 < cols; i += 3){
         if(column[i] == 0) break;
         
         //store in activeParticles the information about the event
         activeParticles.push_back({column[i], column[i+1], column[i+2]});
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


               
         //if the vector of distances is empty only 1 particle remains
         bool isJet = false;
         auto pp = distance_ij.begin();

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

            Particle newParticle = {pT, eta, phi};
            activeParticles.push_back(newParticle);


         }
      }

      std::cout << "Event " << collision << " finished! Found " << Jet.size() << " Jets." << std::endl;

   }

   for (const auto& j : Jet) {
      if (j.pT > 1.0) { 
         output << j.pT << "," << j.eta << "," << j.phi << "\n";
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




