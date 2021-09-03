<!-- mdformat off(b/169948621#comment2) -->

<!--
Semi-automated TOC generation with instructions from
https://github.com/ekalinin/github-markdown-toc#auto-insert-and-update-toc
-->

<!--ts-->
* [Resource Variables](#resource-variables)
   * [API](#api)
   * [Lifecycle](#lifecycle)

<!-- Added by: njeff, at: Thu 02 Sep 2021 02:59:08 PM PDT -->

<!--te-->

# Resource Variables

This doc outlines how to use the TFLite Micro Resource Variables class to use
the VAR_HANDLE, ASSIGN_RESOURCE and READ_RESOURCE operators. This feature is
optional in order to prevent binary bloat on resource constrained systems.

## API

The MicroResourceVariables factory method takes a MicroAllocator and an int
indicating the number of resource varibles to support. This allows the
application to choose the correct number of variables based on the model.

## Lifecycle

When the ResourceVariables class is created in the application, it contains an
array of N ResourceVariable handles. The index into this array is the Resource
ID.

On the first call to Prepare in the VAR_HANDLE op, a new resource ID is reserved
and the resource ID value is referenced from within the output tensor of
VAR_HANDLE. On the first call to Prepare in ASSIGN_VARIABLE, the specified ID
found in the input index tensor is updated based on the size of the input value
tensor, and its resource buffer is allocated.

Future invocations of READ_VARIABLE and ASSIGN_VARIABLE read and write to and
from the allocated resource buffer.

The lifecycle must follow the pattern:
VAR_HANDLE Prepare() -> ASSIGN_VARIABLE Prepare() -> Other calls

Note that VAR_HANDLE Prepare() and ASSIGN_VARIABLE Prepare() may be called more
that once, across multiple subgraphs. Only the first call to each will generate
a new resource ID or allocate a resource buffer.
