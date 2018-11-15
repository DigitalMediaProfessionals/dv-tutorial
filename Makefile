DV_USER_HOME = $(HOME)/dv-sdk/dv-user-driver/

INCLUDE = -ICaffeGoogLeNet/ -Isrc/ -I$(DV_USER_HOME)/include $(shell pkg-config --cflags opencv)
CXXFLAGS = --std=c++11 -Wall -g $(INCLUDE)
LDFLAGS = -L$(DV_USER_HOME)
LIBS    = -ldmpdv $(shell pkg-config --libs opencv)

OBJS = src/dmp_network.o main.o CaffeGoogLeNet/CaffeGoogLeNet_gen.o

TARGET = main

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS) $(LIBS)

clean:
	rm -fr $(TARGET) $(OBJS)
