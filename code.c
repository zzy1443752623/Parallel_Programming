#include <stdio.h>
#include "gputimer.h"
#include "utils.h"


const int N=1024;  
const int K=16;

// to be launched with one thread per element, in (tilesize)x(tilesize) threadblocks
// thread blocks read & write tiles, in coalesced fashion
// adjacent threads read adjacent input elements, write adjacent output elmts
