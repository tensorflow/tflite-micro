# Codegen Hello World Example

This is a code-generated example of the hello world model. The generated source
is checked in for now so that it can be reviewed during the prototyping stage.

## Building the example executable
Please note that this will execute Bazel from make as part of the process.

```
bazel build //codegen/examples/hello_world:hello_world
```

## Running the example

TODO(rjascani): The command works, but it'll just crash as we don't have all of
the data structures fully populated yet.

```
bazel run //codegen/examples/hello_world:hello_world
```

## Updating the generated sources
To update the generated source, you can execute this make target:

```
./codegen/examples/hello_world/update_example_source.sh
```
