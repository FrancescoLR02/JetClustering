#include <iostream>
#include <string>
#include <cmath>
#include <fstream>
#include <vector>
#include <algorithm>
#include <filesystem>
#include <chrono>


const double PI = 3.141592653589793;
const double R = 0.4;
const double maxEta = 5.0;


//Define the grid cells and the total number of cells
int nEta = std::floor((2*maxEta / R));
int nPhi = std::floor((2.0 * PI) / R);
int totCells = nEta * nPhi;


//Structure of partice 
struct Particle{
   int idx;
   double pT, eta, phi;
   bool active;
};


//Functions declaration
void GetCellCoordinates(double eta, double phi, int &i_Eta, int &i_Phi);
int GetIndex(int i_Eta, int i_Phi);
double D_ij(const Particle& p_i, const Particle& p_j);
double D_iB(const Particle& p_i);


int main(int argc, char* argv[]) {

   int numEvents;
   int nColl;

   //Default value
   if (argc < 3) {
      numEvents = 1000;
      nColl = -1;
   }
   else {
      numEvents = std::stod(argv[1]);
      nColl = std::stod(argv[2]);
   }


   //Read input file
   const int cols = 2101;
   const size_t totElements = (size_t)numEvents * cols;

   std::vector<float> data(totElements);

   std::ifstream inFile("data.bin", std::ios::binary);
   if(!inFile){
      std::cerr << "Error opening the file " << std::endl;
      return 1; 
   }

   inFile.read(reinterpret_cast<char*>(data.data()), totElements * sizeof(float));
   inFile.close();


   //Loop of collisions
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
         
         //Fill the activeparticle structure
         activeParticles.push_back({(int)i, (double)ptr[i], (double)ptr[i+1], (double)ptr[i+2], true});
      }

      //Verifying the complexity of the algorithm
      if(nColl != -1){
         if(activeParticles.size() > nColl){
            activeParticles.resize(nColl);
         }
         else continue;
      }

      int activeCount = activeParticles.size();
      while (activeCount > 0){
         
         int numParticles = activeParticles.size();

         //Create the head and next vectors and fill with default values
         std::vector<int> head(totCells, -1);
         std::vector<int> next(numParticles, -1);


         //Fill the vectors 
         for(int i = 0; i < numParticles; ++i){

            //If particle was ereased (not active) continue
            if(!activeParticles[i].active) continue;

            //Get the index for phi and eta in the grid
            int c_Eta, c_Phi;
            GetCellCoordinates(activeParticles[i].eta, activeParticles[i].phi, c_Eta, c_Phi);

            int idx = GetIndex(c_Eta, c_Phi);

            //Fill the heax and next vector
            next[i] = head[idx];
            head[idx] = i;
         }

         //Finding the minimum in a 3x3 cell grid search 
         double minDist = 1e15;
         int best_i = -1, best_j = -1;
         bool isJet = false;

         for(int i = 0; i < activeParticles.size(); ++i){
            if(!activeParticles[i].active) continue;

            //Compute distance form Beam
            double d_iB = D_iB(activeParticles[i]);
            if(d_iB < minDist){
               minDist = d_iB;
               best_i = i;
               best_j = -1;
               isJet = true;
            }

            //Control the nearest neighbohr in the 3x3 grid
            int i_eta, i_phi;
            GetCellCoordinates(activeParticles[i].eta, activeParticles[i].phi, i_eta, i_phi);

            //check in the 3x3
            for(int d_eta = -1; d_eta < 2; d_eta++){
               for(int d_phi = -1; d_phi < 2; d_phi++){

                  //Retrieve the particle in the close cell
                  int n_eta = i_eta + d_eta;
                  
                  // do not consider the edge of the grid
                  if(n_eta < 0 || n_eta >= nEta) continue;

                  //PBC in phi dimension
                  int n_phi = (i_phi + d_phi + nPhi) % nPhi;

                  int cellIdx = GetIndex(n_eta, n_phi);
                  //Retrieve the index from the head cell list
                  int current_j = head[cellIdx];

                  //Until there are particles in the NN cell
                  while (current_j != -1){

                     //This way we don't calculate the same pair twice
                     if(current_j > i && activeParticles[current_j].active){
                        double d_ij = D_ij(activeParticles[i], activeParticles[current_j]);

                        //update distance and best guess for i and j
                        if(d_ij < minDist){
                           minDist = d_ij;
                           best_i = i;
                           best_j = current_j;
                           isJet = false;
                        }
                     }

                     current_j = next[current_j];
                  }
               }
            }
         }

         //if isJet is true, add to Jet structure
         if(isJet){

            //Add the particle to the jet vector
            Jet.push_back(activeParticles[best_i]);

            //deactivate the particle from the list
            activeParticles[best_i].active = false;
            activeCount--;
            

         }

         //else: not a jet, merging the two four-momenta
         else{

            double pT_i = activeParticles[best_i].pT, pT_j = activeParticles[best_j].pT;
            double eta_i = activeParticles[best_i].eta, eta_j = activeParticles[best_j].eta;
            double phi_i = activeParticles[best_i].phi, phi_j = activeParticles[best_j].phi;

            double pT = pT_i + pT_j;
            double eta = (pT_i * eta_i + pT_j * eta_j) / pT;

            //PBC for phi 
            double dPhi = phi_i - phi_j;
            if(dPhi > PI) phi_i -= 2*PI;
            if(dPhi < -PI) phi_i += 2*PI;
            double phi = (pT_i * phi_i + pT_j * phi_j)/pT;

            // Normalize phi back to [-PI, PI]
            if (phi > PI) phi -= 2 * PI;
            if (phi < -PI) phi += 2 * PI;


            // Deactivate old particles
            activeParticles[best_i].active = false;
            activeParticles[best_j].active = false;
            activeCount -= 2;

            // Add the new pseudojet
            Particle newParticle = {-1, pT, eta, phi, true};
            activeParticles.push_back(newParticle);
            activeCount++;


         }

      }

   }
   
   
   return 0;
}


void GetCellCoordinates(double eta, double phi, int &i_Eta, int &i_Phi){

   //Create index for the grid for eta and phi
   i_Eta = std::floor((eta + maxEta) / R);
   i_Phi = std::floor(((phi + PI) / R));

   //clamps
   if(i_Eta < 0) i_Eta = 0;
   if(i_Eta >= nEta) i_Eta = nEta - 1;

   //Periodic boundary condition for phi
   i_Phi = (i_Phi % nPhi + nPhi) % nPhi;

}

int GetIndex(int i_Eta, int i_Phi){
   return i_Eta * nPhi + i_Phi;
}


double D_ij(const Particle& p_i, const Particle& p_j){

   double DeltaPhi = std::abs(p_i.phi - p_j.phi);
   
   //Handles PBC for phi when wraps around the phi cylinder
   if (DeltaPhi > PI) DeltaPhi = 2.0 * PI - DeltaPhi;

   double DeltaR2 = pow(p_i.eta - p_j.eta, 2) + pow(DeltaPhi, 2);
   double d_ij = std::min(pow(p_i.pT, -2), pow(p_j.pT, -2)) * DeltaR2 / pow(R, 2);

   return d_ij;
}

double D_iB(const Particle& p_i){

   double d_iB = pow(p_i.pT, -2);

   return d_iB;
}




