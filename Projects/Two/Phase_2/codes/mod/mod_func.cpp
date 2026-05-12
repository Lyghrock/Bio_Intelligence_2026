#include <stdio.h>
#include "hocdec.h"
#define IMPORT extern __declspec(dllimport)
IMPORT int nrnmpi_myid, nrn_nobanner_;

extern "C" void _CaDynamics_E2_reg();
extern "C" void _Ca_HVA_reg();
extern "C" void _Ca_LVAst_reg();
extern "C" void _Ih_reg();
extern "C" void _Im_reg();
extern "C" void _K_Pst_reg();
extern "C" void _K_Tst_reg();
extern "C" void _NMDA_reg();
extern "C" void _NaTa_t_reg();
extern "C" void _Nap_Et2_reg();
extern "C" void _SK_E2_reg();
extern "C" void _SKv3_1_reg();

extern "C" void modl_reg(){
	//nrn_mswindll_stdio(stdin, stdout, stderr);
    if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
	fprintf(stderr, "Additional mechanisms from files\n");

fprintf(stderr," CaDynamics_E2.mod");
fprintf(stderr," Ca_HVA.mod");
fprintf(stderr," Ca_LVAst.mod");
fprintf(stderr," Ih.mod");
fprintf(stderr," Im.mod");
fprintf(stderr," K_Pst.mod");
fprintf(stderr," K_Tst.mod");
fprintf(stderr," NMDA.mod");
fprintf(stderr," NaTa_t.mod");
fprintf(stderr," Nap_Et2.mod");
fprintf(stderr," SK_E2.mod");
fprintf(stderr," SKv3_1.mod");
fprintf(stderr, "\n");
    }
_CaDynamics_E2_reg();
_Ca_HVA_reg();
_Ca_LVAst_reg();
_Ih_reg();
_Im_reg();
_K_Pst_reg();
_K_Tst_reg();
_NMDA_reg();
_NaTa_t_reg();
_Nap_Et2_reg();
_SK_E2_reg();
_SKv3_1_reg();
}
