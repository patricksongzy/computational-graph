/*
 * MIT License
 *
 * Copyright (c) 2019 Patrick Song
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

package neural.math;

import java.util.Arrays;
import java.util.stream.Collectors;

/**
 * A Tensor represents a n-dimensional matrix of a given node. It can be passed to any node and
 * broadcast to the expected shape.
 */
public class Tensor {

    // the dimensions of the tensor in row-major
    private final int[] dimensions;
    // the length of the dimensions
    private final int length;
    // the values of the tensor
    private float[] values;

    /**
     * Constructs a Tensor from a builder object. This constructor is private, as all tensors must
     * be built from a builder.
     *
     * @param builder the builder object
     * @see Builder
     */
    private Tensor(Builder builder) {
        this.values = builder.values;
        this.dimensions = builder.dimensions;
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

        // the broadcast dimensions is the max of each tensor dimension
        int[] broadcastDimensions = new int[Arrays.stream(tensors).map(Tensor::getDimensions)
            .mapToInt(arr -> arr.length).max().orElseThrow(IllegalArgumentException::new)];
        // the tensor dimensions, padded with ones, where broadcasting is necessary
        int[][] tensorDimensionsPadded = new int[tensors.length][broadcastDimensions.length];

        // loop through the tensors, and copy over the dimensions, leaving the rest as ones
        for (int i = 0; i < tensorDimensionsPadded.length; i++) {
            Arrays.fill(tensorDimensionsPadded[i], 1);
            System.arraycopy(tensors[i].dimensions, 0, tensorDimensionsPadded[i],
                broadcastDimensions.length - tensors[i].dimensions.length,
                tensors[i].dimensions.length);
        }

        for (int i = 1; i <= broadcastDimensions.length; i++) {
            // the broadcast dimension represents the first non-zero dimension encountered, since dimensions which are one can be broadcast
            int j, broadcastDimension = 1;
            for (j = 0; j < tensors.length; j++) {
                broadcastDimension = tensorDimensionsPadded[j][broadcastDimensions.length - i];

                if (broadcastDimension != 1) {
                    break;
                }
            }

            // in order to be broadcast, the remaining dimensions must equal either the broadcast dimension, or one
            for (; j < tensors.length; j++) {
                int currentDimension = tensorDimensionsPadded[j][broadcastDimensions.length - i];

                if (currentDimension != 1 && currentDimension != broadcastDimension) {
                    throw new IllegalArgumentException(String.format(
                        "Unable to broadcast tensors with dimensions '%s'. 1 != '%d' != '%d'.",
                        Arrays.deepToString(tensorDimensionsPadded), currentDimension,
                        broadcastDimension));
                }
            }

            // set the broadcast dimension, since the remaining dimensions are either equal, or one
            broadcastDimensions[broadcastDimensions.length - i] = broadcastDimension;
        }

        // initialize the broadcast tensors to the broadcast dimensions
        Tensor[] broadcast = new Tensor[tensors.length];
        Arrays.setAll(broadcast, t -> Tensor.zeros(broadcastDimensions));

        // loop through the indices of the broadcast tensor
        for (int i = 0; i < broadcast[0].length; i++) {
            // the individual indices for each dimension are calculated from the index in the row-major array
            int[] indices = new int[broadcastDimensions.length];
            // the product of the dimensions, used to calculate the indices for each individual dimension
            int product = 1;

            // the row-major format is as follows:
            // i = r + R * (c + C * (d + D * ...)))
            // this array must be looped backward to ensure the proper dimensions are calculated first
            for (int j = broadcastDimensions.length - 1; j >= 0; j--) {
                // using integer division of the index by the product will result in the index of the dimension
                // taking the remainder will account for further dimensions (a second column may be found on matrices at different depths)
                indices[j] = i / product % broadcastDimensions[j];

                // calculating the product, to ensure that the index of the next dimension is calculated
                product *= broadcastDimensions[j];
            }

            // loop through the tensors to apply the broadcast values
            for (int j = 0; j < tensors.length; j++) {
                // get the values at the index
                // if an index greater than 1 is being accessed on an array broadcast from a dimension of one, that single value is used
                broadcast[j].values[i] = tensors[j]
                    .getBroadcast(tensorDimensionsPadded[j], indices);
            }
        }

        return broadcast;
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
        return Arrays.stream(tensors).map(Tensor::getDimensions)
            .anyMatch(dim -> !Arrays.equals(dim, tensors[0].dimensions));
    }

    /**
     * Constructs a tensor filled with zeros.
     *
     * @param dimensions the dimensions of the tensor
     * @return the tensor filled with zeros
     */
    public static Tensor zeros(int... dimensions) {
        return new Tensor.Builder(dimensions).build();
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
        return Arrays.equals(this.dimensions, other.dimensions) && Arrays
            .equals(this.values, other.values);
    }

    /**
     * Fills the values of the tensor.
     *
     * @param values the values of the tensor.
     */
    @SuppressWarnings("unused")
    public void fill(float[] values) {
        if (values.length != this.values.length) {
            throw new IllegalArgumentException(
                "The dimensions of the inputted values do not match those of the tensor.");
        }

        this.values = values;
    }

    /**
     * Gets a value in the tensor from indices for individual dimensions.
     *
     * @param indices the indices for the individual dimensions
     * @return the value at the given indices
     */
    @SuppressWarnings("WeakerAccess")
    public float get(int... indices) {
        return values[getFlattenedIndex(indices)];
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
    float getBroadcast(int[] dimensionsPadded, int[] indices) {
        int flattenedIndex = indices[0] % dimensionsPadded[0];

        // calculate the flattened (absolute) index, however use the remainders of the broadcast indices and the dimensions
        // this ensures that any broadcast dimensions will have an index of 0, since division by does not result in a remainder
        for (int i = 1; i < dimensionsPadded.length; i++) {
            flattenedIndex =
                flattenedIndex * dimensionsPadded[i] + indices[i] % dimensionsPadded[i];
        }

        return values[flattenedIndex];
    }

    /**
     * Returns the dimensions of the tensor.
     *
     * @return the dimensions of the tensor.
     */
    public int[] getDimensions() {
        return dimensions;
    }

    /**
     * Get the flattened (absolute) index given indices for individual dimensions.
     *
     * @param indices the indices for the individual dimensions
     * @return the flattened (absolute) index
     */
    private int getFlattenedIndex(int... indices) {
        if (!Arrays.stream(indices).allMatch(i -> i >= 0)) {
            throw new IllegalArgumentException("Index cannot be negative.");
        }

        int flattenedIndex = indices[0];
        // i = r + R * (c + C * (d + D * ...))
        for (int i = 1; i < dimensions.length; i++) {
            flattenedIndex = flattenedIndex * dimensions[i] + indices[i];
        }

        return flattenedIndex;
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
    @SuppressWarnings("WeakerAccess")
    public float[] getValues() {
        return values;
    }

    /**
     * Sets a value at given indices for individual dimensions.
     *
     * @param value   the value to set
     * @param indices the indices for the individual dimensions
     */
    @SuppressWarnings("WeakerAccess")
    public void set(float value, int... indices) {
        if (indices.length != dimensions.length) {
            throw new IllegalArgumentException(String
                .format("Indices and dimensions do not match: '%d' != '%d'.", indices.length,
                    dimensions.length));
        }

        values[getFlattenedIndex(indices)] = value;
    }

    /**
     * Sets a value at a given flattened (absolute) index.
     *
     * @param value          the value to set
     * @param flattenedIndex the flattened (absolute) index
     */
    public void set(float value, int flattenedIndex) {
        if (flattenedIndex > length - 1) {
            throw new IllegalArgumentException(String
                .format("Index exceeds length of tensor: '%d' > '%d'", flattenedIndex, length - 1));
        }

        values[flattenedIndex] = value;
    }

    @Override
    public String toString() {
        // print the shape of the tensor
        StringBuilder result = new StringBuilder(String.format("<Tensor: shape=(%s)>",
            Arrays.stream(dimensions).mapToObj(String::valueOf)
                .collect(Collectors.joining(" x "))));

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
         * @param dimensions the dimensions of the tensor
         */
        public Builder(int... dimensions) {
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
            if (values.length != this.length) {
                throw new IllegalArgumentException(String
                    .format("Dimension lengths do not match: '%d', '%d'.", values.length,
                        this.length));
            }

            this.values = values;

            return this;
        }
    }
}
