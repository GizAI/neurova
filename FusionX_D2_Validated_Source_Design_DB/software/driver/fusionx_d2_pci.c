/* Linux PCI driver integration baseline, not a qualified production driver. */
#include <linux/module.h>
#include <linux/pci.h>
#include <linux/dma-mapping.h>
#include "../include/fusionx_d2_abi.h"
#define FX_VENDOR_ID 0x1d1d
#define FX_DEVICE_ID 0xd200
struct fx_dev {struct pci_dev *pdev;void __iomem *bar0;fx_d2_desc_t *sq;dma_addr_t sq_dma;fx_d2_cpl_t *cq;dma_addr_t cq_dma;};
static void fx_wr(struct fx_dev*d,u32 off,u32 v){writel(v,d->bar0+off);}
static int fx_probe(struct pci_dev*pdev,const struct pci_device_id*id){struct fx_dev*d;int rc=pci_enable_device_mem(pdev);if(rc)return rc;rc=dma_set_mask_and_coherent(&pdev->dev,DMA_BIT_MASK(64));if(rc)return rc;rc=pci_request_regions(pdev,"fusionx_d2");if(rc)return rc;pci_set_master(pdev);d=devm_kzalloc(&pdev->dev,sizeof(*d),GFP_KERNEL);if(!d)return -ENOMEM;d->pdev=pdev;d->bar0=pci_iomap(pdev,0,0);if(!d->bar0)return -ENODEV;d->sq=dma_alloc_coherent(&pdev->dev,256*FX_D2_DESCRIPTOR_BYTES,&d->sq_dma,GFP_KERNEL);d->cq=dma_alloc_coherent(&pdev->dev,256*FX_D2_COMPLETION_BYTES,&d->cq_dma,GFP_KERNEL);if(!d->sq||!d->cq)return -ENOMEM;fx_wr(d,FX_D2_REG_SQ_BASE_LO,lower_32_bits(d->sq_dma));fx_wr(d,FX_D2_REG_SQ_BASE_HI,upper_32_bits(d->sq_dma));fx_wr(d,FX_D2_REG_SQ_SIZE,256);fx_wr(d,FX_D2_REG_CQ_BASE_LO,lower_32_bits(d->cq_dma));fx_wr(d,FX_D2_REG_CQ_BASE_HI,upper_32_bits(d->cq_dma));fx_wr(d,FX_D2_REG_CQ_SIZE,256);fx_wr(d,FX_D2_REG_IRQ_MASK,1);fx_wr(d,FX_D2_REG_CONTROL,1);pci_set_drvdata(pdev,d);return 0;}
static void fx_remove(struct pci_dev*pdev){struct fx_dev*d=pci_get_drvdata(pdev);if(!d)return;fx_wr(d,FX_D2_REG_CONTROL,0);if(d->sq)dma_free_coherent(&pdev->dev,256*FX_D2_DESCRIPTOR_BYTES,d->sq,d->sq_dma);if(d->cq)dma_free_coherent(&pdev->dev,256*FX_D2_COMPLETION_BYTES,d->cq,d->cq_dma);if(d->bar0)pci_iounmap(pdev,d->bar0);pci_release_regions(pdev);}
static const struct pci_device_id fx_ids[]={{PCI_DEVICE(FX_VENDOR_ID,FX_DEVICE_ID)},{0}};MODULE_DEVICE_TABLE(pci,fx_ids);static struct pci_driver fx_driver={.name="fusionx_d2",.id_table=fx_ids,.probe=fx_probe,.remove=fx_remove};module_pci_driver(fx_driver);MODULE_LICENSE("GPL");MODULE_DESCRIPTION("FusionX D2 queue/DMA bring-up baseline");
