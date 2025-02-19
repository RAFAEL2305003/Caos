CXX := icpx
CXXFLAGS := -ldnnl -fsycl
SRCS := $(wildcard *.cxx)
OBJS := $(SRCS:.cxx=.o)

all: exec  

final: $(OBJS)
		$(CXX) $(OBJS) -o $@ $(CXXFLAGS)

%.o: %.cxx
		$(CXX) $^ -c -o $@ $(CXXFLAGS)

exec: final
		@./final

rm:
		rm -f $(OBJS)
		rm -i final
