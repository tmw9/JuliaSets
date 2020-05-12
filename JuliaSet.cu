#include "Image.cuh"
#include "Complex.cuh"
#include "parameters.cuh"

#include <iostream>

#include <time.h>

__host__ __device__ int checkPointForJuliaSet(int x, int y) {

    float scaledX = SCALE * (float) (DIM / 2 - x) / (DIM / 2);
    float scaledY = SCALE * (float) (DIM / 2 - y) / (DIM / 2);

    Complex C(REALPARTCOM, COMPLEXPARTCOM);
    Complex Z(scaledX, scaledY);

    int i = 0;
    for(i = 0; i < MAXITER; i++) {
        Z = EQUATION(Z, C);
        if(Z.magnitude2() > MAXMAGNITUDE)
            return 0;
    }
    return 1;
}

__global__ void kernelGPU(Pixel *pixelPtr) {
    int x = blockIdx.x;
    int y = blockIdx.y;

    int offset = x + y * gridDim.x;
    int juliaValue = checkPointForJuliaSet(x, y);
    pixelPtr[offset].r = max(0, min(RED, 255)) * juliaValue;
    pixelPtr[offset].g = max(0, min(GREEN, 255)) * juliaValue;
    pixelPtr[offset].b = max(0, min(BLUE, 255)) * juliaValue;
}

void kernelCPU(Pixel *pixelPtr) {
    int i = 0, j = 0;
    for(j = 0; j < DIM; j++) {
        for(i = 0; i < DIM; i++) {
            int offset = i + j * DIM;
            int juliaValue = checkPointForJuliaSet(i, j);

            // set value for pixels
            pixelPtr[offset].r = max(0, min(RED, 255)) * juliaValue;
            pixelPtr[offset].g = max(0, min(GREEN, 255)) * juliaValue;
            pixelPtr[offset].b = max(0, min(BLUE, 255)) * juliaValue;
        }
    }
}

void printParams(void) {
    std::cout << "DIMENSIONS: " << DIM << " " << DIM << std::endl;
    std::cout << "Max Iterations per pixel: " << MAXITER << std::endl;
    std::cout << "Max Magnitude per pixel: " << MAXMAGNITUDE << std::endl;
}

int main(int argc, char const *argv[]) {
    if(argc < 2) {
        std::cout << "Usage: ./a.out <CPU or GPU>" << std::endl;
        return 0;
    }

    printParams();

    // make clock vars
    clock_t start, end;

    // create image
    Image im(DIM, DIM);
    if(im.createPixels() == false) {
        perror("Error occured");
        return -1;
    }
    if(strcmp(argv[1], "CPU") == 0) {
        start = clock();
        kernelCPU(im.getPixelArrayPointer());
        end = clock();
    } else if(strcmp(argv[1], "GPU") == 0) {
        // GPU Code
        Pixel *gpuPixels;
        cudaMalloc((void **) &gpuPixels, im.getImageSize() * sizeof(Pixel));
        
        dim3 grid(DIM, DIM);
        start = clock();
        kernelGPU<<<grid, 1>>>(gpuPixels);
        end = clock();
        cudaMemcpy(
            im.getPixelArrayPointer(),
            gpuPixels,
            im.getImageSize() * sizeof(Pixel),
            cudaMemcpyDeviceToHost
        );
        cudaFree(gpuPixels);
    } else {
        std::cout << "Invalid option: " << argv[1] << std::endl;
        return 0; 
    }
    
    // print time taken
    double timeTaken = ((double) end - start) / CLOCKS_PER_SEC;
    std::cout << std::fixed << "Time taken by " << argv[1] << ": " << timeTaken  << "s" << std::endl;
    std::cout << "Saving Image..." << std::endl; 
    im.saveImage(FILENAME);

    return 0;
}

