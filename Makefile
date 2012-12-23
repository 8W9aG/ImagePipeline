signsight: clear main.o ImagePipeline.o
	g++ main.o ImagePipeline.o -lopencv_core -lopencv_highgui -lopencv_imgproc -o example

clear:
	rm -f *.o
	rm -f example

main.o:
	g++ -c main.cpp -o main.o

ImagePipeline.o:
	g++ -c ImagePipeline.cpp -o ImagePipeline.o
