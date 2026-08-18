# KVM deployment

AURORA-LLM is a static assembly-only userspace runtime using the Linux syscall ABI. The smallest deployment is therefore a Linux kernel plus an initramfs containing only two files: `/init` (AURORA) and `/model.ali`.

## Build an initramfs appliance

```bash
make
./tools/make_initramfs.sh model.ali aurora-llm.cpio.gz
```

When AURORA is PID 1 it automatically opens `/model.ali`, counts the CPUs in its affinity mask, starts one persistent compute worker per available CPU, and listens on port 8080.

## QEMU/KVM example

Use a guest kernel built with the chosen NIC driver and `CONFIG_IP_PNP=y` built in. Example with virtio-net:

```bash
qemu-system-x86_64 \
  -enable-kvm \
  -cpu host \
  -machine q35,accel=kvm \
  -smp 32,sockets=1,cores=32,threads=1 \
  -m 64G \
  -kernel /path/to/bzImage \
  -initrd aurora-llm.cpio.gz \
  -append 'console=ttyS0 ip=10.0.2.15::10.0.2.2:255.255.255.0::eth0:off' \
  -nic user,model=virtio-net-pci,hostfwd=tcp::8080-:8080 \
  -nographic
```

The model is inside the initramfs in this minimal form. For large production models, use a dedicated guest image or add a direct block-device loader instead of embedding multi-gigabyte weights in initramfs.

## CPU placement

For throughput tests, use physical cores rather than SMT siblings first. Pin the QEMU vCPU threads 1:1 to host physical cores and keep all vCPUs plus memory on one NUMA node where possible. For multi-socket systems, benchmark one model replica per NUMA node before trying cross-socket tensor sharding.

## Huge pages

For a normal Linux guest, back guest RAM with host huge pages using QEMU/libvirt. Huge pages reduce host-side TLB pressure but do not replace model/kernel-level cache optimization inside AURORA.
