CXX = g++
FLAGS = -std=c++11 -O2 -march=native
INCLUDES = -I/usr/local/include/eigen3/
LIBS = -lm

TARGET = main
SRCS = main.cc sparse.cc

OBJS = $(patsubst %.cc,%.o,$(SRCS))

.cc.o:
	$(CXX) $(FLAGS) $(INCLUDES) -c $< -o $@

$(TARGET): $(OBJS)
		$(CXX) $(FLAGS) $(FLAG) -o $@ $(OBJS) $(LIBS)

clean:
	rm $(TARGET) $(OBJS)
