# Configuration

Configuring Core is done via editing a configuration file or using the `core
config` command line tool.

## Prerequisites

- Install core

## Initialize a Core project

To configure anything other than default settings, create a core project, which
will store blackboard configuration information.  The interface is designed to
be similar to git, so hopefully it will feel mostly familiar to many users.

The following shell commands initialize a new core project.

```
mkdir core-bb && cd core-bb
core init
```

This creates a hidden directory with a configuration file in it: `./.core/config.yaml`.

Then, show the default configuration:

```
core config show
```

You can edit the config file directly, or set values using `core config set`
(see below).

**Please note:** these parameters should mostly be left as-is and are
intentionally mostly undocumented for now.

## Setting values

The easiest way to configure parameters is to use `core config set`.

Running
```
core config set object-store.target-gib 4
```

## Running a dedicated core project

To use the new core project configuration, start core inside of the `core-bb`
directory.

```
# Make sure you're in `core-bb` from the example above
core start server
```

## Managing Core's memory utilization

Core's basic configuration uses RAM for its blackboard storage. This is done
because it is fast, and runs anywhere go code runs (no complicated
dependencies).  This helps users get running quickly and on the greatest number
of platforms.

RAM is (sadly) not infinite, so Core employs two strategies to constrain its RAM usage:
- a [ring buffer](https://en.wikipedia.org/wiki/Circular_buffer) 
- [garbage collection](https://en.wikipedia.org/wiki/Garbage_collection_(computer_science)) for the object store

Core's project configuration exposes parameters for these.

```
object-store.gc                15   # determines the tradeoff between garbage collection and memory overshoot
object-store.target-gib        8    # the base storage allocation for the object store
server.size                    8192 # size of the ring buffer (in number of posts)
```

### Object store allocation

**Always start with** the `object-store.target-gib` first.  This is the target
storage use in Gibibytes. Selecting the right size must factor in the amount of
memory you have available on your computer, as well as the amount of data
agents will need to have access to as computations evolve.

For instance, if every message posted has a 1GiB file attached, the default
blackboard configuration would only support 8 messages before garbage
collection would begin, and the earliest objects would become unavailable.  

To keep more messages, we could raise the limit from 8 GiB to 16GiB:

```
core config set object-store.target-gib 16
```

However if your computer has limited RAM (e.g., 16GB), this may not work for
very long.  Finding the correct balance can be task specific.


### The GC parameter

You may notice that Core uses more memory than the base `target-gib` value.
This is set/controlled via the `object-store.gc` parameter.

Our garbage collection process is a less elegant version of Go's.  If you're
interested in learning more about how the GC parameter affects memory
utilization, please check out the GOGC section of Go's [garbage collection
documentation](https://go.dev/doc/gc-guide).

For Core, `object-store.gc` should mostly be left alone. However you can play
around with setting it somewhere between 1 and 100.  I don't recommend setting
it lower than 10. If you set it to 100, then Core will use up to 2x the amount
of target RAM you configure.

### The size of the ring buffer

Once the ring buffer fills, the blackboard will start replacing earlier
messages with the new ones.  This is *separate* from the garbage collection
(which only directly impacts the object store).  

This can have an impact on memory, but should not be your primary vehicle for
tuning memory usage, as it can have substantial knock-on impacts on other parts
of your system.  Please reach out if you have more questions.


