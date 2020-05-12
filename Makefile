JuliaSet: JuliaSet.o Image.o
	nvcc JuliaSet.o Image.o

JuliaSet.o: JuliaSet.cu Image.cuh Complex.cuh parameters.cuh
	nvcc -c JuliaSet.cu 

Image.o: Image.cu Image.cuh
	nvcc -c Image.cu

clean: 
	rm *.o a.out