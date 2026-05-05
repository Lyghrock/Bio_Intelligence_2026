#include <stdio.h>
#include "hocdec.h"
#define IMPORT extern __declspec(dllimport)
IMPORT int nrnmpi_myid, nrn_nobanner_;

extern void _cat1g_reg();
extern void _hh1_reg();

void modl_reg(){
	//nrn_mswindll_stdio(stdin, stdout, stderr);
    if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
	fprintf(stderr, "Additional mechanisms from files\n");

fprintf(stderr," cat1g.mod");
fprintf(stderr," hh1.mod");
fprintf(stderr, "\n");
    }
_cat1g_reg();
_hh1_reg();
}
