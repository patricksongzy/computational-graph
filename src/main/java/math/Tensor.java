/*
 * MIT License
 *
 * Copyright (c) 2020 Patrick Song
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package math;

import graph.node.operation.Operations;
import math.blas.BLAS;
import org.jocl.CL;
import org.jocl.cl_mem;

import java.util.Arrays;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * A Tensor represents a n-dimensional matrix of a given graph.node. It can be passed to any graph.node and
 * broadcasted to the expected shape.
 */
public class Tensor {
    // the dimensions of the tensor in row-major
    private final int[] dimensions;
    // the length of the dimensions
    private final int length;
    // the values of the tensor
    private final float[] values;

    // the GPU buffer for the tensor
    private cl_mem buffer;

    /**
     * Constructs a Tensor from a builder object. This constructor is private, as all tensors must
     * be built from a builder.
     *
     * @param builder the builder object
     * @see Builder
     */
    private Tensor(Builder builder) {
        this.values = builder.values;
        this.dimensions = trimDimensions(builder.dimensions);
        this.length = builder.length;
    }

    /**
     * Broadcasts the given tensors. This alters the tensors such that their shapes are compatible.
     * The dimensions are iterated starting from the number of columns, and are broadcast if the
     * dimensions match, or one dimension equals zero.
     *
     * @param tensors the tensors to broadcast
     * @return an array of broadcast tensors
     */
    public static Tensor[] broadcast(Tensor... tensors) {
        if (tensors.length == 0) {
            throw new IllegalArgumentException("No tensors have been inputted.");
        }

        // the broadcasted dimensions is the max of each tensor dimension
        int[] broadcastedDimensions = new int[Arrays.stream(tensors).map(Tensor::getDimensions).mapToInt(arr -> arr.length).max()
                .orElseThrow(IllegalArgumentException::new)];
        // the tensor dimensions, padded with ones, where broadcasting is necessary
        int[][] tensorDimensionsPadded = new int[tensors.length][broadcastedDimensions.length];

        // loop through the tensors, and copy over the dimensions, leaving the rest as ones
        for (int i = 0; i < tensorDimensionsPadded.length; i++) {
            Arrays.fill(tensorDimensionsPadded[i], 1);
            System.arraycopy(tensors[i].dimensions, 0, tensorDimensionsPadded[i],
                    broadcastedDimensions.length - tensors[i].dimensions.length, tensors[i].dimensions.length);
        }

        for (int i = 1; i <= broadcastedDimensions.length; i++) {
            // the broadcasted dimension represents the first non-zero dimension encountered
            // dimensions which are one can be broadcasted to the new dimension
            int j, broadcastedDimension = 1;
            for (j = 0; j < tensors.length; j++) {
                broadcastedDimension = tensorDimensionsPadded[j][broadcastedDimensions.length - i];

                if (broadcastedDimension != 1) {
                    break;
                }
            }

            // in order to be broadcasted, the remaining dimensions must equal either the broadcasted dimension, or one
            for (; j < tensors.length; j++) {
                int currentDimension = tensorDimensionsPadded[j][broadcastedDimensions.length - i];

                if (currentDimension != 1 && currentDimension != broadcastedDimension) {
                    throw new IllegalArgumentException(
                            String.format("Unable to broadcasted tensors with dimensions '%s'. 1 != '%d' != '%d'.",
                                    Arrays.deepToString(tensorDimensionsPadded), currentDimension, broadcastedDimension));
                }
            }

            // set the broadcasted dimension, since the remaining dimensions are either equal, or one
            broadcastedDimensions[broadcastedDimensions.length - i] = broadcastedDimension;
        }

        // initialize the broadcasted tensors to the broadcasted dimensions
        Tensor[] broadcasted = new Tensor[tensors.length];
        Arrays.setAll(broadcasted, t -> Tensor.zeros(broadcastedDimensions));

        // loop through the indices of the broadcasted tensor
        for (int i = 0; i < broadcasted[0].length; i++) {
            // the individual indices for each dimension are calculated from the index in the row-major array
            int[] indices = getExpandedIndices(broadcastedDimensions, i);

            // loop through the tensors to apply the broadcasted values
            for (int j = 0; j < tensors.length; j++) {
                // get the values at the index
                // if an index greater than 1 is being accessed on an array broadcasted from a dimension of one, that single value is used
                broadcasted[j].values[i] = tensors[j].getBroadcastedValue(tensorDimensionsPadded[j], indices);
            }
        }

        return broadcasted;
    }

    /**
     * Returns the indices for each dimension given a flattened (absolute) index.
     *
     * @param dimensions the dimensions of the tensor
     * @param index      the flattened (absolute) index
     * @return the indices for each dimension
     */
    public static int[] getExpandedIndices(int[] dimensions, int index) {
        // the individual indices for each dimension are calculated from the index in the row-major array
        int[] indices = new int[dimensions.length];
        // the product of the dimensions, used to calculate the indices for each individual dimension
        int product = 1;

        // the row-major format is as follows:
        // i = r + R * (c + C * (d + D * ...)))
        // this array must be looped backward to ensure the proper dimensions are calculated first
        for (int j = dimensions.length - 1; j >= 0; j--) {
            // using integer division of the index by the product will result in the index of the dimension
            // taking the remainder will account for further dimensions (a second column may be found on matrices at different depths)
            indices[j] = index / product % dimensions[j];

            // calculating the product, to ensure that the index of the next dimension is calculated
            product *= dimensions[j];
        }

        return indices;
    }

    /**
     * Get the flattened (absolute) index given indices for individual dimensions.
     *
     * @param dimensions the dimensions of the tensor
     * @param indices    the indices for the individual dimensions
     * @return the flattened (absolute) index
     */
    @SuppressWarnings("WeakerAccess")
    public static int getFlattenedIndex(int[] dimensions, int[] indices) {
        if (Arrays.stream(indices).anyMatch(i -> i < 0))
            throw new IllegalArgumentException("Index cannot be negative.");

        int difference = indices.length - dimensions.length;
        int startIndex = 0, flattenedIndex;

        // check to ensure that leading values are not used
        if (difference > 0)
            startIndex = difference;

        flattenedIndex = indices[startIndex];

        // i = r + R * (c + C * (d + D * ...))
        for (int i = 1; i < dimensions.length; i++) {
            flattenedIndex = flattenedIndex * dimensions[i] + indices[startIndex + i];
        }

        return flattenedIndex;
    }

    /**
     * Checks whether the dimensions of the given tensors match completely.
     *
     * @param tensors the tensors to check
     * @return whether the dimensions of the tensors match completely
     */
    public static boolean isDimensionsMismatch(Tensor... tensors) {
        if (tensors.length == 0) {
            throw new IllegalArgumentException("No tensors have been inputted.");
        }

        // stream the tensors and map to the dimensions to check whether the
        return Arrays.stream(tensors).map(Tensor::getDimensions).anyMatch(dim -> !Arrays.equals(dim, tensors[0].dimensions));
    }

    /**
     * Checks if the given indices are invalid, ignoring leading zeros.
     *
     * @param indices    the indices of the tensor
     * @param dimensions the dimensions of the tensor
     * @return whether the given indices are invalid
     */
    private static boolean isIndicesInvalid(int[] indices, int[] dimensions) {
        if (indices.length < dimensions.length)
            return true;

        if (indices.length > dimensions.length) {
            // extra indices are trimmed
            int difference = indices.length - dimensions.length;

            // ensure the given indices are zeros
            for (int i = 0; i < difference; i++) {
                if (indices[i] != 0)
                    return true;
            }
        }

        return false;
    }

    /**
     * Constructs a tensor filled with ones.
     *
     * @param dimensions the row-major dimensions of the tensor
     * @return the tensor filled with ones
     */
    public static Tensor ones(int... dimensions) {
        Tensor ones = new Tensor.Builder(dimensions).build();
        ones.fill(1);

        return ones;
    }

    /**
     * Trims the given dimensions, removing leading ones.
     *
     * @param dimensions the row-major dimensions of the tensor
     * @return the trimmed row-major dimensions of the tensor
     */
    private static int[] trimDimensions(int[] dimensions) {
        int leadingOnes = 0;
        for (int i = 0; i < dimensions.length; i++) {
            if (dimensions[i] != 1 || i == dimensions.length - 1)
                break;

            leadingOnes++;
        }

        int[] trimmed = new int[dimensions.length - leadingOnes];
        System.arraycopy(dimensions, leadingOnes, trimmed, 0, trimmed.length);

        return trimmed;
    }

    /**
     * Unbroadcasts the given tensor by summing along the broadcasted dimensions.
     *
     * @param tensor                  the tensor to unbroadcast
     * @param unbroadcastedDimensions the row-major dimensions of the unbroadcasted tensor
     * @return the unbroadcasted tensor
     */
    public static Tensor unbroadcast(Tensor tensor, int[] unbroadcastedDimensions) {
        // find the dimensions which have been broadcasted
        int[] broadcastDimensions = IntStream.range(0, tensor.dimensions.length).filter(i -> {
            int broadcastIndex = tensor.dimensions.length - i - 1;
            int unbroadcastIndex = unbroadcastedDimensions.length - i - 1;

            if (unbroadcastIndex < 0)
                return true;

            // if the dimensions mismatch, they have been broadcasted
            return tensor.dimensions[broadcastIndex] != unbroadcastedDimensions[unbroadcastIndex];
        }).map(i -> tensor.dimensions.length - i - 1).toArray();

        if (broadcastDimensions.length == 0)
            return tensor;

        // sum along the broadcasted dimensions
        return Operations.sum(tensor, broadcastDimensions);
    }

    /**
     * Constructs a tensor filled with zeros.
     *
     * @param dimensions the row-major dimensions of the tensor
     * @return the tensor filled with zeros
     */
    public static Tensor zeros(int... dimensions) {
        return new Tensor.Builder(dimensions).build();
    }

    /**
     * Retrieves the memory buffer.
     *
     * @return the memory buffer
     */
    public cl_mem getBuffer() {
        return buffer;
    }

    /**
     * Allocates a memory buffer, and stores it.
     *
     * @param flag the memory buffer flag
     * @return the allocated memory buffer
     */
    public cl_mem allocateBuffer(long flag) {
        buffer = BLAS.allocate(flag, values);
        return buffer;
    }

    /**
     * Updates the values of the tensor from reading its buffer.
     */
    public void readFromBuffer() {
        float[] result = BLAS.readBuffer(buffer, length);
        System.arraycopy(result, 0, values, 0, length);
    }

    /**
     * Updates the memory buffer, releasing the current one if it is already set.
     *
     * @param buffer the memory buffer to set
     */
    public void setBuffer(cl_mem buffer) {
        if (this.buffer != null)
            releaseBuffer();

        this.buffer = buffer;
    }

    public void releaseBuffer() {
        CL.clReleaseMemObject(buffer);
    }

    /**
     * Checks if the dimensions and values of the tensor and another are equal, if the object
     * supplied is a tensor. Returns false if a tensor is not supplied.
     *
     * @param obj the other tensor
     * @return whether the two tensors are equal
     */
    @Override
    public boolean equals(Object obj) {
        if (!(obj instanceof Tensor)) {
            return false;
        }

        Tensor other = (Tensor) obj;
        return Arrays.equals(this.dimensions, other.dimensions) && Arrays.equals(this.values, other.values);
    }

    /**
     * Fills the values of the tensor. If a single values is provided, fills the entire tensor with that value.
     *
     * @param values the values of the tensor.
     */
    @SuppressWarnings({"unused", "WeakerAccess"})
    public void fill(float... values) {
        if (values.length == 1)
            Arrays.fill(this.values, values[0]);
        else if (values.length != this.values.length)
            throw new IllegalArgumentException("The dimensions of the inputted values do not match those of the tensor.");

        System.arraycopy(values, 0, this.values, 0, values.length);
    }

    /**
     * Gets a value in the tensor from indices for individual dimensions.
     *
     * @param indices the indices for the individual dimensions
     * @return the value at the given indices
     */
    @SuppressWarnings("WeakerAccess")
    public float get(int... indices) {
        return values[getFlattenedIndex(dimensions, indices)];
    }

    /**
     * Gets a value at a given flattened (absolute) index.
     *
     * @param flattenedIndex the flattened (absolute) index
     * @return the value at the flattened (absolute) index
     */
    public float get(int flattenedIndex) {
        return values[flattenedIndex];
    }

    /**
     * Gets the value of the tensor at the given indices, accounting for the fact that dimensions of
     * one are broadcast to the required dimensions.
     *
     * @param dimensionsPadded the tensor dimensions, padded with ones where broadcasting is
     *                         necessary
     * @param indices          the indices of the broadcast tensor to access
     * @return the value at the indices, accounting for broadcast dimensions
     */
    float getBroadcastedValue(int[] dimensionsPadded, int[] indices) {
        int flattenedIndex = indices[0] % dimensionsPadded[0];

        // calculate the flattened (absolute) index, however use the remainders of the broadcasted indices and the dimensions
        // this ensures that any broadcasted dimensions will have an index of 0, since division by does not result in a remainder
        for (int i = 1; i < dimensionsPadded.length; i++) {
            flattenedIndex = flattenedIndex * dimensionsPadded[i] + indices[i] % dimensionsPadded[i];
        }

        return values[flattenedIndex];
    }

    /**
     * Returns the row-major dimensions of the tensor.
     *
     * @return the row-major dimensions of the tensor.
     */
    public int[] getDimensions() {
        return dimensions;
    }

    /**
     * Returns the length of the values of the tensor.
     *
     * @return the length of the values of the tensor
     */
    public int getLength() {
        return length;
    }

    /**
     * Returns the float array for the values of the tensor in row-major format.
     *
     * @return the array of values of the tensor
     */
    public float[] getValues() {
        return values;
    }

    public void increment(float value, int... indices) {
        if (isIndicesInvalid(indices, dimensions))
            throw new IllegalArgumentException(
                    String.format("Indices and dimensions do not match: '%d' != '%d'.", indices.length, dimensions.length));

        values[getFlattenedIndex(dimensions, indices)] += value;
    }

    /**
     * Sets a value at given indices for individual dimensions.
     *
     * @param value   the value to set
     * @param indices the indices for the individual dimensions
     */
    @SuppressWarnings("WeakerAccess")
    public void set(float value, int... indices) {
        if (isIndicesInvalid(indices, dimensions)) {
            throw new IllegalArgumentException(
                    String.format("Indices and dimensions do not match: '%d' != '%d'.", indices.length, dimensions.length));
        }

        values[getFlattenedIndex(dimensions, indices)] = value;
    }

    /**
     * Sets a value at a given flattened (absolute) index.
     *
     * @param value          the value to set
     * @param flattenedIndex the flattened (absolute) index
     */
    public void set(float value, int flattenedIndex) {
        if (flattenedIndex > length - 1) {
            throw new IllegalArgumentException(String.format("Index exceeds length of tensor: '%d' > '%d'", flattenedIndex, length - 1));
        }

        values[flattenedIndex] = value;
    }

    @Override
    public String toString() {
        // print the shape of the tensor
        StringBuilder result = new StringBuilder(String.format("<Tensor: shape=(%s)>",
                Arrays.stream(dimensions).mapToObj(String::valueOf).collect(Collectors.joining(" x "))));

        // open the tensor using a bracket
        result.append("\n{");
        // set the default indentation level
        int indentation = 1;

        // loop through the dimensions and add the appropriate bracket and indentation
        for (int i = 0; i < dimensions.length - 1; i++) {
            result.append("\n");

            for (int j = 0; j < indentation; j++) {
                result.append(" ");
            }

            // increase the indentation when opening a bracket
            indentation++;
            result.append("[");
        }

        // loop through the length of the tensor values
        for (int i = 1; i <= length; i++) {
            // check if a row of numbers is being started and indent
            if ((i - 1) % dimensions[dimensions.length - 1] == 0) {
                // start the new row on a new indented line
                result.append("\n");

                for (int j = 0; j < indentation; j++) {
                    result.append(" ");
                }
            }

            result.append(values[i - 1]);

            // check if a row is being ended
            if (i % dimensions[dimensions.length - 1] == 0) {
                // the product of the dimensions, used to calculate the indices for each individual dimension
                int product = dimensions[dimensions.length - 1];

                // the amount of dimensions which are being ended
                int endCount = 0;
                for (int j = dimensions.length - 2; j >= 0; j--) {
                    // check if the current dimension is being ended
                    if (i / product % dimensions[j] == 0) {
                        // the final value of a dimension has been reached
                        endCount++;
                        product *= dimensions[j];
                    } else {
                        // the value no longer belongs to the end of a dimension, therefore will not belong to any further dimensions
                        break;
                    }
                }

                // check if any dimensions have been ended
                if (endCount > 0) {
                    // for each time a dimension is ended, add the bracket and appropriate indentation
                    for (int j = 0; j < endCount; j++) {
                        result.append("\n");

                        indentation--;
                        for (int k = 0; k < indentation; k++) {
                            result.append(" ");
                        }

                        result.append("]");
                    }

                    // begin new dimensions only if the tensor continues
                    if (i != length) {
                        // for each time a dimension is ended, one is started, with the bracket and appropriate indentation
                        for (int j = 0; j < endCount; j++) {
                            result.append("\n");

                            for (int k = 0; k < indentation; k++) {
                                result.append(" ");
                            }

                            indentation++;
                            result.append("[");
                        }
                    }
                }
            } else {
                // if no dimensions are being ended, add a comma and a space
                result.append(", ");
            }
        }

        // close the tensor using a bracket
        result.append("\n}");

        return result.toString();
    }

    /**
     * The Builder builds a tensor, given its dimensions and values.
     */
    public static class Builder {

        private final int[] dimensions;
        private final int length;
        private float[] values;

        /**
         * Constructs the builder, setting the dimensions for the tensor and calculates its length.
         *
         * @param dimensions the row-major dimensions of the tensor
         */
        public Builder(int... dimensions) {
            if (dimensions.length == 0)
                throw new IllegalArgumentException("Cannot have tensor with no dimensions.");

            this.dimensions = dimensions;
            this.length = Arrays.stream(dimensions).reduce(1, (a, b) -> a * b);
        }

        /**
         * Builds the tensor from the supplied dimensions and values. If no values are supplied, the
         * tensor is filled with zeros.
         *
         * @return the tensor
         */
        public Tensor build() {
            if (values == null) {
                this.values = new float[length];
            }

            return new Tensor(this);
        }

        /**
         * Sets the values of the tensor.
         *
         * @param values the values of the tensor
         * @return this builder
         */
        public Builder setValues(float... values) {
            if (values.length < this.length) {
                throw new IllegalArgumentException(
                        String.format("Dimension lengths do not match: '%d', '%d'.", values.length, this.length));
            }

            this.values = values;

            return this;
        }
    }
}
