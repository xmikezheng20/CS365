# OSX compiler
CC = clang++

# Dwarf compiler
CC = g++

CXX = $(CC)

CFLAGS = -I/usr/local/include/opencv4 -I../include -std=c++11
# OSX include paths (for MacPorts)
#CFLAGS = -I/opt/local/include -I../include

# OSX include paths (for homebrew, probably)
#CFLAGS = -I/usr/local/include -I../include

# Dwarf include paths
#CFLAGS = -I../include # opencv includes are in /usr/include
CXXFLAGS = $(CFLAGS)

LDFLAGS = -L/usr/local/lib
# OSX Library paths (if you use MacPorts)
#LDFLAGS = -L/opt/local/lib

#OSX Library paths (if you use homebrew, probably)
#LDFLAGS = -L/usr/local/lib

# Dwarf Library paths
#LDFLAGS = -L/usr/lib/x86_64-linux-gnu # opencv libraries are here

# opencv libraries
LDLIBS = -lopencv_core -lopencv_highgui -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_ml


BINDIR = ../bin

# dependencies
IDIR = ../include

_DEPS = processing.hpp classifier.hpp
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

OBJ_objRec = objRec.o processing.o classifier.o

objRec: $(OBJ_objRec)
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

vidCapture: vidCapture.o
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

classifierTest: classifierTest.o classifier.o
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

clean:
	rm -f *.o *~
