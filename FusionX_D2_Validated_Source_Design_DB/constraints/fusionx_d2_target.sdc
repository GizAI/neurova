# Architectural target only; this file is not evidence of timing closure.
create_clock -name core_clk -period 1.000 [get_ports core_clk]
create_clock -name host_clk -period 4.000 [get_ports host_clk]
set_clock_groups -asynchronous -group [get_clocks core_clk] -group [get_clocks host_clk]
set_false_path -from [get_ports {release_compute_host thermal_ok_host hbm_init_host hbm_fatal_host}] -to [get_cells -hierarchical *u_status_sync*q1*]
