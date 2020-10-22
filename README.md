# computational-graph
[Documentation](https://patricksongzy.github.io/computational-graph)

A GPU-parallel Java automatic differentiation computational graph implementation using OpenCL linear algebra bindings.

## Example
```java
Constant a = new Constant(new Tensor.Builder().setDimensions(2, 3).setValues(3, 8, 2, 5, 1, 6).build());
Constant b = new Constant(new Tensor.Builder().setDimensions(1, 3).setValues(3, 2, 1).build());

Multiplication c = new Multiplication(a, b);

Graph.compute(c);
Graph.gradient();

assertThat(Results.getGradient(a)).isEqualTo(new Tensor.Builder().setDimensions(2, 3).setValues(3, 2, 1, 3, 2, 1).build());
assertThat(Results.getGradient(b)).isEqualTo(new Tensor.Builder().setDimensions(1, 3).setValues(8, 9, 8).build());
```

## Improvements
Make initialization of tensor values more streamlined.
```java
import static math.Tensor.vec;
// using generics, this should be possible
Tensor tensor = vec(vec(1, 2),
                    vec(2, 3),
                    vec(4, 5));
```