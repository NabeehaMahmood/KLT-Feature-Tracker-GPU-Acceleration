######################################################################
# Choose your favorite C compiler
CC = gcc
CXX = g++

######################################################################
# -DNDEBUG prevents the assert() statements from being included in 
# the code.  If you are having problems running the code, you might 
# want to comment this line to see if an assert() statement fires.
FLAG1 = -DNDEBUG

######################################################################
# -DKLT_USE_QSORT forces the code to use the standard qsort() 
# routine.  Otherwise it will use a quicksort routine that takes
# advantage of our specific data structure to greatly reduce the
# running time on some machines.  Uncomment this line if for some
# reason you are unhappy with the special routine.
# FLAG2 = -DKLT_USE_QSORT

######################################################################
# Add your favorite C flags here.
CFLAGS = $(FLAG1) $(FLAG2) -O3 -g
CXXFLAGS = $(CFLAGS)

######################################################################
# Data directory path
DATA_DIR = ../../data


######################################################################
# There should be no need to modify anything below this line (but
# feel free to if you want).

EXAMPLES = example1.c example2.c example3.c example4.c example5.c
ARCH = convolve.c error.c pnmio.c pyramid.c selectGoodFeatures.c \
       storeFeatures.c trackFeatures.c klt.c klt_util.c writeFeatures.c
LIB = -L/usr/local/lib -L/usr/lib

.SUFFIXES:  .c .o .cpp

all:  lib klt_app $(EXAMPLES:.c=)

.c.o:
	$(CC) -c $(CFLAGS) $<

.cpp.o:
	$(CXX) -c $(CXXFLAGS) $<

lib: $(ARCH:.c=.o)
	rm -f libklt.a
	ar ruv libklt.a $(ARCH:.c=.o)
	rm -f *.o

# Main application with C++ main
klt_app: lib main.o
	$(CXX) $(CXXFLAGS) -o $@ main.o -L. -lklt $(LIB) -lm

example1: libklt.a
	$(CC) $(CFLAGS) -o $@ $@.c -L. -lklt $(LIB) -lm

example2: libklt.a
	$(CC) $(CFLAGS) -o $@ $@.c -L. -lklt $(LIB) -lm

example3: libklt.a
	$(CC) $(CFLAGS) -o $@ $@.c -L. -lklt $(LIB) -lm

example4: libklt.a
	$(CC) $(CFLAGS) -o $@ $@.c -L. -lklt $(LIB) -lm

example5: libklt.a
	$(CC) $(CFLAGS) -o $@ $@.c -L. -lklt $(LIB) -lm

# Profiling targets
profile: lib klt_app
	@echo "Running profiling with gprof..."
	$(CXX) $(CXXFLAGS) -pg -o klt_app_prof main.o -L. -lklt $(LIB) -lm
	./klt_app_prof
	gprof klt_app_prof gmon.out > profile_report.txt
	@echo "Profile report saved to profile_report.txt"

# Benchmark target
benchmark: klt_app
	@echo "Running performance benchmark..."
	@for i in 1 2 3 4 5; do \
		echo "Running example $$i:"; \
		time ./klt_app; \
	done

# Copy data files to current directory for testing
setup_data:
	cp $(DATA_DIR)/*.pgm .

depend:
	makedepend $(ARCH) $(EXAMPLES) main.cpp

clean:
	rm -f *.o *.a $(EXAMPLES:.c=) klt_app klt_app_prof *.tar *.tar.gz libklt.a \
	      feat*.ppm features.ft features.txt gmon.out profile_report.txt *.pgm

help:
	@echo "KLT Feature Tracker V1 - Available targets:"
	@echo "  all        - Build library and main application"
	@echo "  lib        - Build KLT library"
	@echo "  klt_app    - Build main application"
	@echo "  example1-5 - Build individual examples"
	@echo "  profile    - Run with profiling enabled"
	@echo "  benchmark  - Run performance benchmark"
	@echo "  setup_data - Copy data files to current directory"
	@echo "  clean      - Remove generated files"
	@echo "  help       - Show this help message"



