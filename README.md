# Introduction

This artifact supports the paper *Controlling Adaptive HARQ Erasure Coding for Real-Time Transport under Channel Model Mismatch* (ECRTS 2026) and contains the Jupyter notebook evaluation scripts.

Running the complete suite of experiments to reproduce all paper results takes approximately 1 hour on the specified hardware.


# Quick-start

The following sections contain the native quick-start guide.

## System Requirements
To faithfully reproduce the results of the paper, we recommend evaluating on at least 32 GB of ram (DDR4 or DDR5), and 8 CPU cores (AMD Zen 3+, Intel 12th gen+). We recommend avoiding big/little CPU core architectures, and thus recommend AMD Zen 3,4,5 processors, or Intel Gen 12+ CPUs with efficiency/little cores disabled. The evaluation code targets the Linux operating system. The artifact can be evaluated on any modern Linux OS running kernel 6.12+ with the following Real-time Kernel Requirements. We recommend using Debian 13 (Trixie) and installing the linux-image-rt-amd64 kernel package (see Native Quick-start `VARIANT 1`).

## Getting-started Checklist
1. If you have less than 24 GB of ram, you need to edit `prrt/prrt/src/constants.rs` and lower `TRACE_DURATION` from `240` seconds to `180` seconds, or at the lowest `130` seconds for the evaluation to run. This should reduce the memory usage to roughly 10GB for `130` seconds, and 13GB for `180` seconds.
2. If you run on old/slow hardware or use virtualization, you might run into the issue that your kernel networking stack starts dropping packets, which might lead to congestion collapse in the evaluation. Make sure to increase `SLOWDOWN_FACTOR` in the convergence or mismatch notebooks, which slows down the packets per second, to whatever your hardware can sustain (See the notebooks for more information).

## Native Quick-start
This Quick-start is written specifically for Debian 13.4. Most parts of this quick-start will work on debian-based distributions, except for `VARIANT 1` of the  Real-time kernel capabilities section, as the `linux-image-rt-amd64` package is debian-specific.

Debian 13.4 specific:
```bash
su -
usermod -aG sudo your_username
su - your_username
```

Install Basic Tools
```bash
sudo apt update
sudo apt install build-essential git curl
# Enable netem kernel module, we use tc-netem in the evaluation
sudo modprobe sch_netem
echo "sch_netem" | sudo tee -a /etc/modules # make it persistent across reboots
# sudo apt install libcap2-bin # should be already installed on debian 13, used for `sudo setcap`
```

Enabling Real-time kernel capabilities of Linux systems
```bash
# VARIANT 1: install the full RT kernel (Below for Debian 13, you may need to adapt this for other distributions)
sudo apt update
sudo apt install linux-image-rt-amd64
# Afterwards, you need to reboot to use the new RT kernel.
sudo reboot

# VARIANT 2: Enable Full Dynamic Preemption on the stock kernel
# PRECONDITION: Verify your current kernel supports dynamic preemption.
# The command below should output "CONFIG_PREEMPT_DYNAMIC=y". 
# If it outputs nothing or "=n", your kernel does not support this and you must use VARIANT 1.
grep CONFIG_PREEMPT_DYNAMIC /boot/config-$(uname -r)

# 1. Open the GRUB configuration file using a text editor (e.g., nano)
sudo nano /etc/default/grub

# 2. Locate the line that begins with GRUB_CMDLINE_LINUX_DEFAULT
# 3. Append "preempt=full" inside the quotation marks. 
#    For example, change:
#    GRUB_CMDLINE_LINUX_DEFAULT="quiet"
#    To:
#    GRUB_CMDLINE_LINUX_DEFAULT="quiet preempt=full"

# 4. Save the file and exit the editor (in nano: Ctrl+O, Enter, Ctrl+X)

# 5. Update GRUB to apply the new configuration
sudo update-grub

# 6. Reboot the system to load the new kernel parameters
sudo reboot

# 7. To verify this worked, check the currently active dynamic preemption model.
# The following command should now say "full"
sudo dmesg | grep "Dynamic Preempt"
# Alternatively, verify that the boot parameter (preempt=full) was successfully passed to the kernel:
cat /proc/cmdline

# To undo these changes:
# Variant 1:
sudo apt install linux-image-amd64 # ensure the default kernel exist and wasnt purged for some reason.
sudo apt remove --purge linux-image-rt-amd64 # you can safely remove the rt kernel, the running one is not affected.
sudo apt autoremove --purge
sudo update-grub
sudo reboot
uname -v # to verify, this should not contain "PREEMPT_RT"
# Variant 2:
sudo nano /etc/default/grub # remove " preempt=full"
sudo update-grub
sudo reboot
```

First, get the source code, including the `prrt` submodule that contains the PRRT protocol implementation
```bash
git clone https://github.com/miodic/prrt-eval-artifacts.git prrt-eval
cd prrt-eval
# pulls prrt from https://github.com/miodic/prrt.git
git submodule update --init --recursive
```

Next, we have to get the evaluation code running, you can follow these steps:
```bash
# 1. Install rust (you can skip this if you have rustc >= 1.88.0 installed), we used 1.92 to produce our evaluation plots, this artifact does not strictly need this exact version, msrv should be 1.88.0
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- --default-toolchain 1.92.0
. "$HOME/.cargo/env" # note the leading .
# or
source "$HOME/.cargo/env"

# 2. Python setup, we use `uv` packet manager (its fast!)
# Or use your own python environment manager
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# 3. Set up the virtual environment
uv python install 3.13 # installs into ~/.local/share/uv/python/
cd python
uv venv --python 3.13 # creates .venv/
uv pip install -r requirements.txt
# you dont need to enter the venv, you can simply let uv take care of it:
# uv run your_script.py
```

Finally, to get the evaluation scripts to run
```bash
# 1. build the parameterized evalation binary
# This uses RT-thread priorities (cap_sys_nice=eip) and linux traffic control `tc` (cap_net_admin)
# IMPORTANT: rerun this if you change the source code in src/main.rs or prrt/ to see the effects in the evaluation notebooks.
cd ..
cargo build --bin main --release && sudo setcap 'cap_net_admin,cap_sys_nice=eip' ./target/release/main

# 2. Run the notebook
# from within prrt-eval/python
cd python
uv run jupyter lab
# or
uv run jupyter notebook
# You can disable authentication tokens with
uv run jupyter lab --ServerApp.token='' --ServerApp.password=''
```

> ⚠️ **Important for Rebuilding:**
> ``Linux capabilities are silently stripped every time the binary is rebuilt. If you modify the code and re-run `cargo build`, you **MUST** immediately re-run the `sudo setcap` command. Otherwise, the evaluation will crash with a corresponding missing capability error. We use setcap to avoid running the binary with `sudo`.``

The notebooks defines an experiment which consists of network loss and delay configurations and prrt application parameters. The parameterized `src/main.rs` parses these inputs, instanciates the experiment and simulates it.
This generates traces, which the notebooks visualize. The experiments run on the `loopback` device, and losses/delays are introduced with linux traffic control (tc/netem).

## Docker Quick-start
We assume docker to be installed on your system. Follow the below command to load and run the artifact.

Import and run the docker image
```bash
# Load the provided docker image
sudo docker load -i prrt_ecrts_artifact_docker_240sTraces.tar

# And then run it with the following command:
sudo docker run -it --cap-add=NET_ADMIN --cap-add=SYS_NICE -p 8888:8888 prrt-ecrts26-artifact

# Finally, open your browser and navigate to localhost:8888 to open juypter lab
firefox 127.0.0.1:8888
```

## OVA Quick-start
The OVA browser has a bookmark that points to our evaluation repository, previewing this `README.md`.

Open a terminal and follow these commands to get started. The artifact is already set up, which simplifies the setup phase.
```bash
cd prrt-eval

# Start Jupyter lab
cd python
uv run jupyter lab
```

## Steps to generate PRRT_ecrts_artifact.tar yourself (Not needed if you use one of the provided images)
```bash
git submodule update --init # manually pull the prrt/ subdirectory
sudo docker build -t prrt-ecrts26-artifact .
sudo docker save prrt-ecrts26-artifact > prrt_ecrts_artifact.tar
```

# Reproducing the Results

Once you have set up the environment and started Jupyter Lab, navigate to the `python/` directory and run the following notebooks to reproduce specific results:

* **Figure 2:** with `eval_convergence.ipynb`.
* **Figure 3:** with `eval_missmatch.ipynb`.
* **Figure 4:** with `incremental_search_eval.ipynb`.
* **Table 2:** with `execution_time_CDF.ipynb`.

The `eval_missmatch.ipynb` and `eval_convergence.ipynb` notebooks define experiments via network configurations (loss rate, delay) and PRRT application parameters. They then call the parameterized `src/main.rs` binary to instantiate and simulate the experiment, generating traces that the notebook then visualizes.  
The `incremental_search_eval.ipynb` and `execution_time_CDF.ipynb` notebooks evaluate the incremental search via the `inc_search_eval/src/main.rs` harness.

# Repository organization
   - `prrt/` contains the workspace (groups of library crates) of the PRRT reference implementation used in the evaluations
   - `prrt/prrt/` contains the main library that implements the PRRT reference implementation.
   - `prrt/prrt-bin/src/\{sender.rs,receiver.rs\}` implement a minimal, sender and receiver application as a standalone demo of the protocol. Consult `prrt/README.md` for how to run them.
   - `src/main.rs` is the parameterized evaluation binary that the Jupyter notebooks build and call to evaluate PRRT.
   - `inc\_search\_eval/` is a rust crate that evaluates the PRRT incremental search and its non-greedy variant.
   - `python/` contains the python scripts in form of Jupyter notebooks to reproduce the figures of the paper.
   - `python/traces/` contains the data that was used to generate the paper Figure 2 and Figure 3.
   - `python/inc_search_eval/paper_data/` contains the data that was used to generate the paper Figure 4 and Table 2.
   - `Dockerfile` can be used to build a docker image to shorten the setup phase (alternatively use the images we provide below)

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.