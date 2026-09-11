#include <stdint.h>
#include "fusionx_d2_abi.h"
typedef struct {volatile uint32_t *bar;uint32_t secure_ok,hbm_ok,thermal_ok;} fx_fw_t;
int fx_fw_release_compute(fx_fw_t *fw){if(!fw||!fw->bar)return -1;if(!(fw->secure_ok&&fw->hbm_ok&&fw->thermal_ok))return -2;fw->bar[FX_D2_REG_CONTROL/4]=1;return 0;}
void fx_fw_mask_irqs(fx_fw_t *fw){if(fw&&fw->bar)fw->bar[FX_D2_REG_IRQ_MASK/4]=0;}
