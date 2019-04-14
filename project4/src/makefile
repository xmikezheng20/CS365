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
LDLIBS = -lopencv_core -lopencv_highgui -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_calib3d
LDLIBS_2 = -lopencv_core -lopencv_highgui -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_calib3d -lopencv_aruco


BINDIR = ../bin

# # dependencies
# IDIR = ../include
#
# _DEPS = processing.hpp classifier.hpp
# DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

OBJ_calib = calib.o
OBJ_augmented = augmented.o shape.o
OBJ_ar_img = ar_img.o shape.o
OBJ_ar_harris = ar_harris.o shape.o
OBJ_ar_harris_detect = ar_harris_detect.o shape.o
OBJ_ar_aruco = ar_aruco.o shape.o

calib: $(OBJ_calib)
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

augmented: $(OBJ_augmented)
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

ar_img: $(OBJ_ar_img)
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

ar_harris_detect: $(OBJ_ar_harris_detect)
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

ar_harris: $(OBJ_ar_harris)
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS)

ar_aruco: $(OBJ_ar_aruco)
	$(CC) $^ -o $(BINDIR)/$@ $(LDFLAGS) $(LDLIBS_2)

clean:
	rm -f *.o *~
