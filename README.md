# PubHunt
_Hunt for Bitcoin public keys._

## It searches random compressed public keys for given hash160.

#
# The idea to do this

This is only useful for Bitcoin [puzzle transaction](https://www.blockchain.com/btc/tx/08389f34c98c606322740c0be6a7125d9860bb8d5cb182c02f98461e5fa6cd15).

Modified to Search for the public key within a given range.

For the puzzles, ```66```, ```67```, ```68```, ```69```, ```71``` or ```72``` and some more, there are no public keys are available, so if we can able to find public keys for those addresses, then [Pollard Kangaroo](https://github.com/JeanLucPons/Kangaroo) algorithm can be used to solve those puzzles.

That's it, cheers 🍺 

# Usage

This is GPU only, no CPU support. 

```
./PubHunt -h
PubHunt [-check] [-h] [-v]
        [-gi GPU ids: 0,1...] [-gx gridsize: g0x,g0y,g1x,g1y, ...]
        [-o outputfile] [-sr startRange] [-er endRange] [inputFile]

 -v                       : Print version
 -gi gpuId1,gpuId2,...    : List of GPU(s) to use, default is 0
 -gx g1x,g1y,g2x,g2y, ... : Specify GPU(s) kernel gridsize, default is 8*(MP number),128
 -o outputfile            : Output results to the specified file
 -sr startRange           : Start of the key range to search
 -er endRange             : End of the key range to search
 -cp checkpointFile       : Checkpoint file for crash recovery (default pubhunt.checkpoint)
 -cpt seconds             : Seconds between checkpoints, 0 disables (default 300)
 -l                       : List cuda enabled devices
 -check                   : Check Int calculations
 inputFile                : List of the hash160, one per line in hex format (text mode)
```

### Checkpoints & crash recovery

PubHunt performs a **random** (Monte-Carlo) search: cuRAND draws random keys and
maps them into `[startRange, endRange]`, so there is no sequential position to
resume from. Instead the program persists its cumulative progress to a
checkpoint file every 5 minutes (configurable via `-cpt`) and on a clean exit
(Ctrl-C). If the program is killed unexpectedly, the next launch with the same
range and input file automatically:

- restores the accumulated key count, elapsed time and found counter, and
- advances the quasi-random sequence past the keys already scanned, so coverage
  continues instead of restarting the same region.

Found keys are always appended to the output file the instant they are found, so
they are never lost regardless of checkpointing. The checkpoint is written
atomically (temp file + rename) and is ignored if its stored range / hash160 set
does not match the current run. Use `-cpt 0` to disable checkpointing.

For example:
```
./PubHunt -sr 2000000000000000 -er 3fffffffffffffff -o KeyFound.txt -gi 0 -gx 8192,1024 hash160.txt

PubHunt v1.00

DEVICE       : GPU
GPU IDS      : 0
GPU GRIDSIZE : 8192x1024
NUM HASH160  : 1
OUTPUT FILE  : KeyFound.txt
KEY RANGE    : 0x2000000000000000 - 0x3fffffffffffffff
GPU          : GPU #0 NVIDIA GeForce RTX 4070 (46x0 cores) Grid(8192x1024)

[00:00:14] [GPU: 2681.56 MH/s] [T: 37,547,409,408 (36 bit)] [F: 0]
```

### Performance notes

- The GPU kernel is pure hashing (SHA-256 + RIPEMD-160), so throughput is
  bound by integer/hash rate. The status line reports **hash operations** (two
  per candidate X, for the even and odd public keys), i.e. the distinct-key rate
  is about half the printed `MK/s`.
- The RNG generation is double-buffered so cuRAND overlaps with the compute
  kernel, and the oversized per-thread stack reservation inherited from
  VanitySearch has been removed.
- If throughput looks low, first check the GPU is not power/thermal throttling
  (`nvidia-smi -l 1`: clocks, power, temperature), then sweep the grid size —
  e.g. `-gx 8704,128`, `-gx 4352,256`, or omit `-gx` for the default. Very large
  thread-per-group values (like `1024`) often lower occupancy.

## Building

##### GitHub Actions (Windows 10/11 x64, universal CUDA build)

A ready-to-use workflow is included at
[`.github/workflows/build-windows.yml`](.github/workflows/build-windows.yml).
It installs the CUDA Toolkit, builds a **universal fat binary** (SASS for
Maxwell → Hopper plus PTX for forward JIT) and uploads `PubHunt.exe` as a build
artifact. The binary runs on a **GeForce RTX 3080 (Ampere, compute capability
8.6)** and most other modern NVIDIA GPUs.

To use it:
1. Push this fork to GitHub (the workflow runs on every push, or run it manually
   from the **Actions** tab → *Build Windows (CUDA)* → *Run workflow*).
2. Download the `PubHunt-win64-cuda...` artifact from the finished run.
3. Pushing a tag like `v1.0` also publishes the binary as a GitHub Release.

The set of GPU architectures is defined by `CodeGeneration` in
[`PubHunt/PubHunt.vcxproj`](PubHunt/PubHunt.vcxproj); add or remove `sm_XX`
entries there to trim the binary. To build for **only** the RTX 3080 (smaller,
faster to compile) reduce it to `compute_86,sm_86;compute_86,compute_86`.

##### Windows (local)
- Microsoft Visual Studio Community 2022 (toolset v143)
- CUDA Toolkit 11.1+ (12.x recommended for RTX 3080 / sm_86)
- Open `PubHunt.sln`, select `Release | x64`, and build.

##### Linux
 - Edit the makefile and set up the appropriate CUDA SDK and compiler paths for nvcc. Or pass them as variables to `make` command.

    ```make
    CUDA       = /usr/local/cuda
    CXXCUDA    = /usr/bin/g++
    ```
 - To build with CUDA: pass CCAP value according to your GPU compute capability
   (RTX 3080 = `86`)
    ```sh
    $ make CCAP=86 all
    ```

## License
PubHunt is licensed under GPLv3.

## Disclaimer
ALL THE CODES, PROGRAM AND INFORMATION ARE FOR EDUCATIONAL PURPOSES ONLY. USE IT AT YOUR OWN RISK. THE DEVELOPER WILL NOT BE RESPONSIBLE FOR ANY LOSS, DAMAGE OR CLAIM ARISING FROM USING THIS PROGRAM.
