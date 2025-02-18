CXX := icpx
CXXFLAGS := -ldnnl
SRCS := $(wildcard *.cxx)
OBJS := $(SRCS:.cxx=.o)

all: exec  

final: $(OBJS)
		$(CXX) $(OBJS) -o $@ $(CXXFLAGS)

%.o: %.cxx
		$(CXX) $^ -c -o $@

exec: 
		@./final

rm:
		rm -i $(OBJS)
		rm -i final
