all:
	g++ -I /usr/local/cuda/include cudak_cpu.cpp -o exec -lcuda `wx-config --cppflags` `wx-config --libs`
	nvcc -I/usr/local/cuda/include -arch sm_20 -ptx cudak_gpu.cu -o __cudak_gpu.ptx

clean:
	rm exec __cudak_gpu.ptx
