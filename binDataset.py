import h5py
import hdf5plugin

def main():
   #Load the file
   print('Opening the file...')
   with h5py.File("events_anomalydetection_Z_XY_qqq.h5", 'r') as f:
      # Load the dataframe with the data
      data = f['df/block0_values'][:] 

   data.astype('float32').tofile("data.bin")
   print(f"Saved {data.shape} matrix to data.bin")

if __name__ == "__main__":
   main()