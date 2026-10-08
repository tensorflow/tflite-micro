---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!--
This markdown file is the source of truth for this tutorial. The paired
mnist_compression_tutorial.ipynb is generated from it, because Google Colab
can only open .ipynb files. After editing this file, regenerate the notebook
by running the bazel target :ipynb.update and commit both files.
The bazel test :ipynb_test fails if the two files
are out of sync.

Notes for editors (this file is in the jupytext markdown format):
- Code cells are ```python fenced blocks; everything else is markdown.
- Two consecutive blank lines split adjacent markdown cells.
-->

# TFLM Compression Tutorial: MNIST [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/compression/tutorial/mnist_compression_tutorial.ipynb)

This tutorial demonstrates TensorFlow Lite for Microcontrollers (TFLM) model compression
using the MNIST dataset.

## Beyond Standard Quantization: Compression Enabled by Weight Clustering

While quantization (converting float32 to INT8) is standard practice in
embedded ML and provides a reliable 4x size reduction, TFLM can achieve even
better results through compression of weight-clustered models. To unlock this
additional compression, you'll need to perform two key steps: weight clustering
during model training, followed by LUT-based compression during post-processing.

### Why Weight Clustering is Essential

Neural network weights are typically unique values spread across a continuous
range. This diversity makes them nearly impossible to compress effectively - there's no
pattern or redundancy to exploit. Weight clustering changes this fundamentally by:

- Grouping similar weights into a limited number of clusters
- Replacing diverse weight values with a small set of shared values
- Creating the redundancy that compression algorithms need to work

Without clustering, compression algorithms struggle because every weight is different.
With clustering, we can represent each weight using just an index into a small lookup table.
The number of clusters (and thus the compression ratio) is a parameter you control based
on your accuracy requirements.

## The Complete Pipeline

TFLM combines three techniques:

1. **Weight Clustering** (the enabler): Groups weights into clusters, creating patterns
   that can be compressed. Without this step, meaningful compression isn't possible.

2. **Quantization** (standard practice): Converts float32 to INT8, providing the baseline
   4x reduction that's common in embedded deployments.

3. **Look-Up Table (LUT) Compression** (the payoff): Leverages the clustered structure to
   store indices and a lookup table, achieving compression beyond what quantization alone
   can provide.

## Current Limitations

As of today, not every operator in TFLM supports compression. This is actively being
improved, and this tutorial will be updated as support expands. Currently, compression
works with fully connected and convolutional layers.

## What You'll Learn

In this tutorial, you'll:
- Train a simple CNN model for MNIST digit classification
- Apply weight clustering to specific layers using TensorFlow Model Optimization toolkit
- Convert and quantize the model for embedded deployment
- Apply TFLM's LUT compression for maximum size reduction
- Evaluate the tradeoffs between compression ratio and accuracy

For more details on weight clustering, refer to:
- [TensorFlow Model Optimization Guide](https://www.tensorflow.org/model_optimization/guide/clustering)
- [Clustering Example](https://www.tensorflow.org/model_optimization/guide/clustering/clustering_example)
- [Comprehensive Clustering Guide](https://www.tensorflow.org/model_optimization/guide/clustering/clustering_comprehensive_guide)


## Import Required Libraries

First, we'll import all the necessary libraries for this tutorial. We need:
- TensorFlow for model training
- TFLM Python package for compression and embedded inference simulation
- TensorFlow Model Optimization toolkit for weight clustering
- NumPy and Matplotlib for data handling and visualization

```python
# Install the tflite-micro package if it isn't already available in this
# notebook's environment, e.g. on the first run in Google Colab or in a
# fresh local environment.
import importlib.util
if importlib.util.find_spec("tflite_micro") is None:
    %pip install tflite-micro

import tensorflow as tf
import tflite_micro as tflm
import numpy as np
import matplotlib.pyplot as plt

# Import tf_keras for compatibility. This is a standalone Keras implementation that
# maintains stable APIs across TensorFlow versions. Using tf_keras instead of tf.keras
# ensures that the TensorFlow Model Optimization toolkit (used for weight clustering)
# works consistently regardless of the TensorFlow version installed.
import tf_keras

print(f"TensorFlow version: {tf.__version__}")
print(f"tf_keras version: {tf_keras.__version__}")
print(f"TFLM module: {tflm.__version__}")
```

## Load and Prepare MNIST Dataset

MNIST is a classic dataset of handwritten digits (0-9). Each image is 28x28
pixels in grayscale. We'll use this simple dataset to show the
effects of compression without the complexity of larger models.

The data preparation steps are standard:
- Normalize pixel values to [0, 1] range.
- Add a channel dimension since convolutional layers expect 3D input (height, width, channels)

```python
# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Define the input shape for MNIST images
# This will be used throughout the tutorial for model creation and TFLite conversion
# Shape format: [batch_size, height, width, channels]
# - None: Variable batch size (TFLite will use batch size 1 at inference)
# - 28, 28: MNIST image dimensions
# - 1: Grayscale images have 1 channel
MNIST_INPUT_SHAPE = [None, 28, 28, 1]

# Normalize pixel values to [0, 1]
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# Add channel dimension
train_images = train_images[..., np.newaxis]
test_images = test_images[..., np.newaxis]

print(f"Training images shape: {train_images.shape}")
print(f"Training labels shape: {train_labels.shape}")
print(f"Test images shape: {test_images.shape}")
print(f"Test labels shape: {test_labels.shape}")
```

## Visualize Sample Images

Let's visualize some sample images to understand our dataset. This helps verify
that our data is loaded correctly and gives us intuition about the
classification task.

```python
# Display sample images
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
axes = axes.ravel()

for i in range(10):
    axes[i].imshow(train_images[i].squeeze(), cmap='gray')
    axes[i].set_title(f'Label: {train_labels[i]}')
    axes[i].axis('off')

plt.suptitle('Sample MNIST Images')
plt.tight_layout()
plt.show()
```

## Create a Model Architecture for Compression Demonstration

For this tutorial, we'll use a model that's appropriately sized to demonstrate
compression while keeping training time reasonable.

TFLM currently supports compression on:
- **Conv2D layers**: Convolutional weights can be compressed
- **Dense layers**: Converted to FullyConnected operators in TFLite, also compressible

```python
# Create a model architecture optimized for demonstrating compression
model = tf_keras.Sequential([
    tf_keras.layers.Input(shape=MNIST_INPUT_SHAPE[1:]),  # Skip batch dimension
    
    # Convolutional layers
    tf_keras.layers.Conv2D(16, (5, 5), activation='relu'),
    tf_keras.layers.MaxPooling2D((2, 2)),
    
    # Second conv layer
    tf_keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf_keras.layers.MaxPooling2D((2, 2)),
    
    # Flatten for fully connected layers
    tf_keras.layers.Flatten(),
    
    # Fully connected layers
    tf_keras.layers.Dense(128, activation='relu'),
    tf_keras.layers.Dense(64, activation='relu'),
    
    # Output layer
    tf_keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
```

## Train the Model

Now we'll train our baseline model. Note that this tutorial focuses on demonstrating
compression techniques, not optimal training practices. We're using 3 epochs which
is sufficient for MNIST and keeps the tutorial quick.

For production training best practices, see:
- [TensorFlow Training Guide](https://www.tensorflow.org/guide/keras/training_with_built_in_methods)
- [Keras Training Documentation](https://keras.io/guides/training_with_built_in_methods/)

Pay attention to the final test accuracy - we'll use this to measure how much
accuracy we trade for compression.

```python
# Train the model (simplified training for demonstration)
history = model.fit(
    train_images, train_labels,
    epochs=3,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
print(f"\nTest accuracy: {test_accuracy:.4f}")
```

## Plot Training History

Visualizing the training history helps us verify that our model trained properly.
We should see decreasing loss and increasing accuracy over epochs, with validation
metrics following similar trends (indicating no overfitting).

```python
# Plot training history
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history.history['loss'], label='Training Loss')
ax1.plot(history.history['val_loss'], label='Validation Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation Loss')
ax1.legend()
ax1.grid(True)

ax2.plot(history.history['accuracy'], label='Training Accuracy')
ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Training and Validation Accuracy')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
```

## Apply Weight Clustering to Selected Layers

Everything we've done so far - loading data, creating a model,
training it - is standard deep learning workflow that you'd use for any neural network.
From this point forward, we enter compression-specific territory with weight clustering,
the critical first step that enables TFLM's compression pipeline.

### Why Clustering Enables Compression

Without clustering, neural network weights are typically unique floating-point values
spread across a continuous range. This makes compression nearly impossible because
there's no redundancy to exploit. Weight clustering groups similar weights together,
creating redundancy that compression algorithms can leverage.

For example, if we cluster weights into 16 groups:
- Original: thousands of unique 32-bit float values
- After clustering: only 16 unique values
- Result: each weight can be represented by a 4-bit index into a 16-entry lookup table

### Selective Layer Clustering

We don't need to cluster every layer. In fact, selective clustering often yields better
results because:
- Some layers (like the final classification layer) might be more sensitive to clustering
- Different layers might benefit from different numbers of clusters
- You can balance model size and accuracy by choosing which layers to compress

In this tutorial, we'll cluster only the second Conv2D layer and the
first fully-connected layer. This demonstrates targeted compression while
maintaining the original precision in other layers.

### Implementation Details

We use TensorFlow Model Optimization toolkit's clustering API, which:
- Replaces weights with cluster centroids during training
- Fine-tunes the model to adapt to clustered weights
- Maintains cluster assignments while updating centroids

For comprehensive documentation on clustering techniques, see:
- [TensorFlow Model Optimization Clustering Guide](https://www.tensorflow.org/model_optimization/guide/clustering)

```python
# Install the TensorFlow Model Optimization toolkit if it isn't already
# available in this notebook's environment.
import importlib.util
if importlib.util.find_spec("tensorflow_model_optimization") is None:
    %pip install tensorflow_model_optimization

import tensorflow_model_optimization as tfmot

# Define clustering parameters
cluster_weights = tfmot.clustering.keras.cluster_weights
CentroidInitialization = tfmot.clustering.keras.CentroidInitialization

print("Applying weight clustering to selected layers...")

# First, we need to clone the original model to preserve it
model_config = model.get_config()
cloned_model = tf_keras.Sequential.from_config(model_config)
cloned_model.set_weights(model.get_weights())

# Define which layers to cluster and with how many clusters
# We'll target the larger layers for maximum compression impact
# 
# Note: We're storing this configuration in CLUSTERING_CONFIG because we'll need
# to reference it later when specifying which layers should be compressed during
# the TFLite conversion process. This ensures consistency between the layers we
# cluster and the layers we compress.
CLUSTERING_CONFIG = {
    'conv2d_1': 16,  # Second conv layer
    'dense': 16,     # First dense layer
}

# Important: The cluster_weights() function doesn't modify layers in-place.
# Instead, it returns a wrapper layer that adds clustering functionality.
# Therefore, we need to:
# 1. Walk through each layer
# 2. Wrap layers that need clustering
# 3. Collect all layers (wrapped and unwrapped)
# 4. Build a new Sequential model from the collected layers

clustered_layers = []
for layer in cloned_model.layers:
    if layer.name in CLUSTERING_CONFIG:
        # Wrap this layer with clustering functionality
        num_clusters = CLUSTERING_CONFIG[layer.name]
        clustered_layer = cluster_weights(
            layer,
            number_of_clusters=num_clusters,
            cluster_centroids_init=CentroidInitialization.KMEANS_PLUS_PLUS
        )
        clustered_layers.append(clustered_layer)
        print(f"Applied {num_clusters} clusters to layer: {layer.name}")
    else:
        # Keep the original layer unchanged
        clustered_layers.append(layer)

# Create a new Sequential model from our list of layers
# (This is necessary because Sequential models require rebuilding when modifying layers)
clustered_model = tf_keras.Sequential(clustered_layers)

print(f"Clustered model created")
print("\nClustered layers:")
for layer in clustered_model.layers:
    if 'cluster' in str(type(layer)).lower():
        print(f"  - {layer.name}: {type(layer).__name__}")
    else:
        print(f"  - {layer.name}: not clustered")

# Compile the clustered model
clustered_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Build the model by calling it on dummy data
clustered_model.build(input_shape=MNIST_INPUT_SHAPE)

print("Clustered model summary:")
clustered_model.summary()

# Fine-tune the clustered model
# Note: We use 2 epochs for demonstration. For optimal fine-tuning strategies, see:
# https://www.tensorflow.org/model_optimization/guide/clustering/clustering_comprehensive_guide
print("\nFine-tuning clustered model...")
clustered_history = clustered_model.fit(
    train_images, train_labels,
    epochs=2,  # Simplified fine-tuning for demonstration
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

# Evaluate clustered model
clustered_test_loss, clustered_test_accuracy = clustered_model.evaluate(test_images, test_labels, verbose=0)
print(f"\nClustered model test accuracy: {clustered_test_accuracy:.4f}")

# Strip clustering wrappers to get the final model
final_clustered_model = tfmot.clustering.keras.strip_clustering(clustered_model)

# Compare model sizes before conversion
print("\nModel comparison:")
print(f"Original model parameters: {model.count_params():,}")
print(f"Clustered model parameters: {final_clustered_model.count_params():,}")
```

## Convert Models to TFLite

Now we convert our models to TensorFlow Lite format, required by the TFLM
interpreter. We follow a two-step approach that's standard in TensorFlow Lite
workflows:

### Why Convert to Float First, Then Quantize?

You might wonder why we don't quantize directly during conversion. This two-step 
approach (float → quantized) is actually the preferred workflow for several reasons:

1. **Debugging and Validation**: The float model serves as a baseline for accuracy
   comparison. You can verify the conversion works correctly before adding quantization
   complexity.

2. **Flexibility**: You can experiment with different quantization strategies on the
   same float model without re-converting from Keras/TensorFlow each time.

3. **Gradual Optimization**: This approach lets you measure the impact of each
   optimization step separately - first conversion, then quantization, then compression.

### What This Step Accomplishes

Converting to TFLite format:
- Provides a standardized model representation that TFLM can work with
- Optimizes the model graph for mobile and embedded deployment
- Allows us to measure the baseline model size

At this stage, you'll notice that clustering alone hasn't significantly reduced
the model size. This is expected! The real size reduction comes when we combine
clustering with quantization and TFLM's LUT compression in the following steps.

```python
# Convert the original model to TFLite format using concrete function
#
# Why use get_concrete_function()?
# - Using concrete functions ensures the resulting model will work with TFLM
# - Keras models are high-level abstractions that support multiple input shapes
# - TFLM requires all tensor shapes to be known at compile time (no dynamic tensors)
# - When converting Keras models directly via convert_from_keras(), the TFLite
#   converter sometimes creates models with dynamic tensors that are
#   incompatible with TFLM
#
# The conversion process:
# 1. Wrap the Keras model in tf.function to create a TensorFlow graph
# 2. Call get_concrete_function() with a TensorSpec that defines the exact input shape
# 3. This creates a "concrete function"---a graph with all shapes and types determined
# 4. TFLite converter can then convert this concrete graph
concrete_func = tf.function(lambda x: model(x)).get_concrete_function(
    tf.TensorSpec(shape=MNIST_INPUT_SHAPE, dtype=tf.float32)
)
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
tflite_model = converter.convert()

# Save the model
with open('mnist_model.tflite', 'wb') as f:
    f.write(tflite_model)

print(f"Original model size: {len(tflite_model):,} bytes ({len(tflite_model)/1024:.2f} KB)")

# Convert the clustered model to TFLite format using concrete function
# Same process as above---we need to create a concrete function with fixed input shapes
clustered_concrete_func = tf.function(lambda x: final_clustered_model(x)).get_concrete_function(
    tf.TensorSpec(shape=MNIST_INPUT_SHAPE, dtype=tf.float32)
)
converter_clustered = tf.lite.TFLiteConverter.from_concrete_functions([clustered_concrete_func])
tflite_clustered_model = converter_clustered.convert()

# Save the clustered model
with open('mnist_model_clustered.tflite', 'wb') as f:
    f.write(tflite_clustered_model)

print(f"Clustered model size: {len(tflite_clustered_model):,} bytes ({len(tflite_clustered_model)/1024:.2f} KB)")
print(f"Size reduction from clustering: {(1 - len(tflite_clustered_model)/len(tflite_model))*100:.1f}%")
```

## Apply Post-Training Quantization

Next, we apply post-training quantization to convert our models from float32 to INT8.
This is standard practice in TensorFlow Lite deployment - virtually all production
models use quantization for the benefits it provides:

- **4x size reduction**: 32-bit floats → 8-bit integers
- **Faster inference**: Integer operations are more efficient on microcontrollers
- **Lower power consumption**: Integer math requires less energy

### This is Not the Novel Part

To be clear: quantization is a form of compression, but it's not what makes this
tutorial special. INT8 quantization is routine in embedded ML deployment. What's
unique here is how our earlier clustering step sets up the model for TFLM's
LUT-based compression in the next phase.

### Quantization's Role in Our Pipeline

While quantization doesn't enable compression per se, it plays an important role
in our pipeline:
- TFLM's LUT compression currently operates on integer models
- Clustered weights naturally map well to INT8 representation
- The overall pipeline (clustering → quantization → LUT compression) works together
  to achieve the final size reduction

So while this quantization step is standard practice, it's a necessary part of
the complete compression workflow.

```python
# Provide representative dataset for full integer quantization
def representative_dataset():
    for i in range(100):
        # Get a random batch of input data
        indices = np.random.randint(0, len(train_images), size=1)
        yield [train_images[indices].astype(np.float32)]

# Convert original model with post-training quantization using concrete function
# Even though we're quantizing, the concrete function still expects float32 input
# The quantization happens during conversion, not in the concrete function definition
quant_concrete_func = tf.function(lambda x: model(x)).get_concrete_function(
    tf.TensorSpec(shape=MNIST_INPUT_SHAPE, dtype=tf.float32)
)
converter = tf.lite.TFLiteConverter.from_concrete_functions([quant_concrete_func])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

quantized_model = converter.convert()

# Save the quantized model
with open('mnist_model_quantized.tflite', 'wb') as f:
    f.write(quantized_model)

print("Original model with quantization:")
print(f"  Size: {len(quantized_model):,} bytes ({len(quantized_model)/1024:.2f} KB)")
print(f"  Compression ratio: {len(tflite_model) / len(quantized_model):.2f}x")
print(f"  Size reduction: {(1 - len(quantized_model)/len(tflite_model))*100:.1f}%")

# Convert clustered model with post-training quantization using concrete function
clustered_quant_concrete_func = tf.function(lambda x: final_clustered_model(x)).get_concrete_function(
    tf.TensorSpec(shape=MNIST_INPUT_SHAPE, dtype=tf.float32)
)
converter_clustered_quant = tf.lite.TFLiteConverter.from_concrete_functions([clustered_quant_concrete_func])
converter_clustered_quant.optimizations = [tf.lite.Optimize.DEFAULT]
converter_clustered_quant.representative_dataset = representative_dataset
converter_clustered_quant.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter_clustered_quant.inference_input_type = tf.uint8
converter_clustered_quant.inference_output_type = tf.uint8

clustered_quantized_model = converter_clustered_quant.convert()

# Save the clustered+quantized model
with open('mnist_model_clustered_quantized.tflite', 'wb') as f:
    f.write(clustered_quantized_model)

print("\nClustered model with quantization:")
print(f"  Size: {len(clustered_quantized_model):,} bytes ({len(clustered_quantized_model)/1024:.2f} KB)")
print(f"  Compression ratio: {len(tflite_model) / len(clustered_quantized_model):.2f}x")
print(f"  Size reduction: {(1 - len(clustered_quantized_model)/len(tflite_model))*100:.1f}%")

# Summary of all model sizes
print("\n" + "="*50)
print("MODEL SIZE SUMMARY:")
print("="*50)
print(f"Original float32 model:         {len(tflite_model):,} bytes")
print(f"Clustered float32 model:        {len(tflite_clustered_model):,} bytes")
print(f"Quantized int8 model:           {len(quantized_model):,} bytes")
print(f"Clustered + Quantized model:    {len(clustered_quantized_model):,} bytes")
print(f"\nBest compression ratio: {len(tflite_model) / len(clustered_quantized_model):.2f}x")
```

## Inspect Model Architecture and Data Types

In production workflows, you might use sophisticated tools to visualize model graphs,
such as Netron (interactive model visualizer), TensorBoard, or TFLite's built-in
HTML visualization. However, for this tutorial, simple text-based output provides
everything we need to understand our small MNIST model. The text output clearly shows
which layers have been quantized, their quantization parameters, and tensor
shapes - sufficient for verifying that quantization worked correctly without
additional dependencies.

```python
def print_model_details(model_content, model_name="Model"):
    """Print detailed information about a TFLite model including data types."""
    # For model inspection, we'll use TFLite since TFLM doesn't expose tensor details
    # TFLM is designed for inference, not model introspection
    interpreter = tf.lite.Interpreter(model_content=model_content)
    interpreter.allocate_tensors()
    
    print(f"\n{model_name} Details:")
    print("="*50)
    
    # Get tensor details
    tensor_details = interpreter.get_tensor_details()
    
    # Print input details
    input_details = interpreter.get_input_details()
    print("\nINPUT TENSORS:")
    for inp in input_details:
        tensor = tensor_details[inp['index']]
        print(f"  Name: {tensor['name']}")
        print(f"  Shape: {tensor['shape']}")
        print(f"  Type: {tensor['dtype']}")
        if 'quantization' in tensor and tensor['quantization'][0] != 0:
            print(f"  Quantization: scale={tensor['quantization'][0]:.6f}, zero_point={tensor['quantization'][1]}")
        print()
    
    # Print output details
    output_details = interpreter.get_output_details()
    print("OUTPUT TENSORS:")
    for out in output_details:
        tensor = tensor_details[out['index']]
        print(f"  Name: {tensor['name']}")
        print(f"  Shape: {tensor['shape']}")
        print(f"  Type: {tensor['dtype']}")
        if 'quantization' in tensor and tensor['quantization'][0] != 0:
            print(f"  Quantization: scale={tensor['quantization'][0]:.6f}, zero_point={tensor['quantization'][1]}")
        print()
    
    # Print layer information
    print("ALL LAYERS:")
    print(f"{'Index':<6} {'Name':<40} {'Type':<10} {'Shape':<20} {'Quantization':<30}")
    print("-"*110)
    
    for i, tensor in enumerate(tensor_details):
        quant_info = ""
        if 'quantization' in tensor and tensor['quantization'][0] != 0:
            quant_info = f"s={tensor['quantization'][0]:.4f}, zp={tensor['quantization'][1]}"
        
        print(f"{i:<6} {tensor['name'][:39]:<40} {str(tensor['dtype']):<10} {str(tensor['shape']):<20} {quant_info:<30}")

# Print details for both models
print_model_details(tflite_model, "Original Float32 Model")
print_model_details(quantized_model, "Quantized INT8 Model")
```

## Compare Model Accuracy

Now we evaluate how compression techniques affect model accuracy. This is a critical
step because it quantifies the accuracy-size tradeoff.

We use TFLM's Python interpreter to simulate how these models will perform on
microcontrollers. This gives us confidence that our accuracy measurements reflect
real-world embedded performance.

Key points to observe:
- Quantization typically causes a small accuracy drop
- Clustering might add minimal additional accuracy loss
- The combined techniques usually maintain acceptable accuracy
- If accuracy drops too much, consider using more clusters or compressing fewer layers

```python
# Helper function to evaluate TFLite model using TFLM
def evaluate_tflite_model(model_content, test_images, test_labels):
    # Use TFLM interpreter
    interpreter = tflm.runtime.Interpreter.from_bytes(bytes(model_content))
    
    # TFLM uses different API - get details for index 0
    INPUT_INDEX = 0
    OUTPUT_INDEX = 0
    input_details = interpreter.get_input_details(INPUT_INDEX)
    output_details = interpreter.get_output_details(OUTPUT_INDEX)
    
    correct = 0
    total = min(1000, len(test_images))  # Evaluate on subset for speed
    
    for i in range(total):
        # Prepare input
        test_image = test_images[i:i+1]
        
        # Quantize input if needed
        if input_details['dtype'] == np.uint8:
            quant_params = input_details.get('quantization_parameters', {})
            if 'scales' in quant_params and 'zero_points' in quant_params:
                input_scale = quant_params['scales'][0]
                input_zero_point = quant_params['zero_points'][0]
                test_image = test_image / input_scale + input_zero_point
                test_image = test_image.astype(np.uint8)
        
        # Run inference using TFLM API
        interpreter.set_input(test_image, INPUT_INDEX)
        interpreter.invoke()
        
        # Get output
        output = interpreter.get_output(OUTPUT_INDEX)[0]
        
        # Dequantize output if needed
        if output_details['dtype'] == np.uint8:
            quant_params = output_details.get('quantization_parameters', {})
            if 'scales' in quant_params and 'zero_points' in quant_params:
                output_scale = quant_params['scales'][0]
                output_zero_point = quant_params['zero_points'][0]
                output = (output.astype(np.float32) - output_zero_point) * output_scale
        
        predicted = np.argmax(output)
        if predicted == test_labels[i]:
            correct += 1
    
    return correct / total

# Evaluate all models
print("Evaluating models on test subset...")
original_accuracy = evaluate_tflite_model(tflite_model, test_images, test_labels)
clustered_accuracy = evaluate_tflite_model(tflite_clustered_model, test_images, test_labels)
quantized_accuracy = evaluate_tflite_model(quantized_model, test_images, test_labels)
clustered_quantized_accuracy = evaluate_tflite_model(clustered_quantized_model, test_images, test_labels)

print("\n" + "="*50)
print("ACCURACY COMPARISON:")
print("="*50)
print(f"Original float32 model:      {original_accuracy:.4f}")
print(f"Clustered float32 model:     {clustered_accuracy:.4f}")
print(f"Quantized int8 model:        {quantized_accuracy:.4f}")
print(f"Clustered + Quantized model: {clustered_quantized_accuracy:.4f}")
print(f"\nAccuracy drop from clustering+quantization: {(original_accuracy - clustered_quantized_accuracy)*100:.2f}%")
```

## Apply TFLM Compression to the Clustered Layer

This is where the magic happens! TFLM's Look-Up Table (LUT) compression leverages
the clustering we applied earlier to achieve further size reduction.

### How LUT Compression Works

Remember our clustered layers have only 16 unique weight values. Instead of storing
each weight as an 8-bit integer (after quantization), TFLM can:
1. Store a table of the 16 unique values (the "look-up table")
2. Replace each weight with a 4-bit index into this table
3. Result: 2x additional compression (8 bits → 4 bits per weight)

### The Process

To apply TFLM compression, we need to:
1. Identify the specific tensor (layer weights) to compress
2. Create a compression specification
3. Apply the compression transformation

### Finding the Right Tensor

TFLite models store weights as separate tensors connected to operations. We need
to trace through the model graph to find the weight tensor for our target layer.
This requires some understanding of TFLite's model structure.

```python
# Import TFLM compression module
from tflite_micro import compression
from tensorflow.lite.python import schema_py_generated as schema_fb

def find_weight_tensor_for_layer(model_bytes, layer_name):
    """Find weight tensor for a specific layer by following the operation graph.
    
    This works by:
    1. Finding the operation that contains the layer name in its output
    2. Following the operation's inputs to find the weight tensor
    """
    buf = bytearray(model_bytes)
    model = schema_fb.Model.GetRootAsModel(buf, 0)
    subgraph = model.Subgraphs(0)
    
    # Find the operation for this layer
    for i in range(subgraph.OperatorsLength()):
        op = subgraph.Operators(i)
        
        # Check output tensor names
        for output_idx in range(op.OutputsLength()):
            tensor_idx = op.Outputs(output_idx)
            tensor = subgraph.Tensors(tensor_idx)
            tensor_name = tensor.Name().decode('utf-8')
            
            if layer_name in tensor_name and ('Conv2D' in tensor_name or 'MatMul' in tensor_name):
                # Found the operation! Now get its weight input
                # For Conv2D and Dense layers: input[0]=activation, input[1]=weights, input[2]=bias
                if op.InputsLength() >= 2:
                    weight_tensor_idx = op.Inputs(1)
                    weight_tensor = subgraph.Tensors(weight_tensor_idx)
                    
                    # Get tensor info
                    shape = [weight_tensor.Shape(i) for i in range(weight_tensor.ShapeLength())]
                    name = weight_tensor.Name().decode('utf-8')
                    
                    return {
                        'index': weight_tensor_idx,
                        'name': name,
                        'shape': shape
                    }
    
    return None

# Find tensors to compress for all clustered layers
print("Finding weight tensors for compressed layers...")
tensors_to_compress = []

for layer_name in CLUSTERING_CONFIG.keys():
    weight_info = find_weight_tensor_for_layer(clustered_quantized_model, layer_name)
    
    if weight_info:
        print(f"\nFound {layer_name} weight tensor:")
        print(f"  Index: {weight_info['index']}")
        print(f"  Name: '{weight_info['name']}'")
        print(f"  Shape: {weight_info['shape']}")
        tensors_to_compress.append(weight_info)
    else:
        print(f"Warning: Could not find weight tensor for layer {layer_name}")

if not tensors_to_compress:
    raise ValueError("Could not find any weight tensors to compress")
```

## Create and Apply Compression Specification

Now we create the compression specification that tells TFLM exactly how to compress
our model. TFLM provides two ways to specify compression:

### Programmatic API (Used Here)
The SpecBuilder API allows us to programmatically define compression:
- `add_tensor()`: Specifies which tensor to compress
- `with_lut()`: Configures Look-Up Table compression
- `index_bitwidth=4`: Uses 4-bit indices (perfect for our 16 clusters)

### YAML Configuration (Alternative)

In production workflows, the clustering step might be performed by a separate tool
or pipeline that analyzes which tensors to compress and with what parameters. That
tool would then write a YAML specification file describing the compression strategy.
This decouples the compression specification from the compression execution:

```yaml
tensors:
  - subgraph: 0
    tensor: 7  # conv2d_1 weights
    compression:
      lut:
        index_bitwidth: 4
  - subgraph: 0
    tensor: 9  # dense weights
    compression:
      lut:
        index_bitwidth: 4
```

### Compression Impact

The overall compression ratio depends on several factors:

1. **Number of layers compressed**: The more layers you compress, the closer you 
   approach the theoretical maximum compression ratio. For example, with INT8 
   quantization (8-bit values) and 16 clusters (4-bit indices), you can approach 
   50% compression (8 bits → 4 bits) as more layers are compressed.

2. **Size of compressed layers**: Larger layers contribute more to the overall 
   compression. Compressing a 256-unit dense layer has more impact than compressing 
   a small convolutional layer.

3. **Lookup table overhead**: Each compressed tensor needs a lookup table (16 × 8 
   bits = 128 bits for 16 clusters). This overhead is negligible for large tensors 
   but can be significant for very small ones.

4. **Model architecture**: Models with many large fully-connected or convolutional 
   layers benefit more from compression than models dominated by small layers or 
   non-compressible operations.

In this tutorial, we're compressing only 2 out of 8 layers, so our overall 
compression ratio is less than the theoretical maximum. If we compressed all 
eligible layers, we could achieve closer to 2x compression on top of quantization.

```python
print("\nCreating compression specification...")
# Build compression spec for all identified tensors
spec_builder = compression.SpecBuilder()

for tensor_info in tensors_to_compress:
    (spec_builder
         .add_tensor(subgraph=0, tensor=tensor_info['index'])
             .with_lut(index_bitwidth=4))
    print(f"Added tensor {tensor_info['index']} ({tensor_info['name']}) to compression spec")

compression_spec = spec_builder.build()

print(f"\nCompression spec created for {len(tensors_to_compress)} tensors")
print("  Compression type: Look-Up Table (LUT)")
print("  Index bitwidth: 4 bits (supports up to 16 unique values)")

# Apply TFLM compression
print("\nApplying TFLM compression...")
try:
    compressed_model = compression.compress(clustered_quantized_model, compression_spec)
    
    # Save the compressed model
    with open('mnist_model_tflm_compressed.tflite', 'wb') as f:
        f.write(compressed_model)
    
    print(f"\nTFLM compressed model size: {len(compressed_model):,} bytes ({len(compressed_model)/1024:.2f} KB)")
    print(f"Additional compression from TFLM: {(1 - len(compressed_model)/len(clustered_quantized_model))*100:.1f}%")
    print(f"Total compression vs original: {len(tflite_model) / len(compressed_model):.2f}x")
    
    # Update the size summary
    print("\n" + "="*50)
    print("UPDATED MODEL SIZE SUMMARY:")
    print("="*50)
    print(f"Original float32 model:         {len(tflite_model):,} bytes")
    print(f"Clustered float32 model:        {len(tflite_clustered_model):,} bytes")
    print(f"Quantized int8 model:           {len(quantized_model):,} bytes")
    print(f"Clustered + Quantized model:    {len(clustered_quantized_model):,} bytes")
    print(f"TFLM Compressed model:          {len(compressed_model):,} bytes")
    print(f"\nBest compression ratio: {len(tflite_model) / len(compressed_model):.2f}x")
    
except Exception as e:
    print(f"Compression failed: {e}")
    print("This might happen if the tensor doesn't have enough unique values for effective compression")
    compressed_model = None
```

## Verify TFLM Compressed Model Performance

The final step is to verify that our compressed model still performs well. This is
crucial because we want to ensure that the compression didn't degrade the model's
functionality.

### What to Look For
- The compressed model should maintain similar accuracy to the clustered+quantized version
- Any additional accuracy drop should be minimal
- If accuracy drops significantly, consider:
  - Using more clusters (e.g., 32 instead of 16)
  - Compressing different or fewer layers
  - Fine-tuning the clustered model for more epochs

### Deployment Considerations
Remember that this compressed model will run on microcontrollers using TFLM's C++
inference engine, which includes support for decompressing LUT-compressed tensors during
inference.

```python
if compressed_model is not None:
    # Evaluate the TFLM compressed model
    print("Evaluating TFLM compressed model...")
    tflm_compressed_accuracy = evaluate_tflite_model(compressed_model, test_images, test_labels)
    
    print("\n" + "="*50)
    print("FINAL ACCURACY COMPARISON:")
    print("="*50)
    print(f"Original float32 model:      {original_accuracy:.4f}")
    print(f"Clustered float32 model:     {clustered_accuracy:.4f}")
    print(f"Quantized int8 model:        {quantized_accuracy:.4f}")
    print(f"Clustered + Quantized model: {clustered_quantized_accuracy:.4f}")
    print(f"TFLM Compressed model:       {tflm_compressed_accuracy:.4f}")
    print(f"\nAccuracy drop from full compression: {(original_accuracy - tflm_compressed_accuracy)*100:.2f}%")
```

## Visualize Predictions

Let's visualize some actual predictions to see how our compressed model performs
compared to the original.

```python
# Compare predictions from both models
def get_predictions(model_content, images):
    # Use TFLM interpreter
    interpreter = tflm.runtime.Interpreter.from_bytes(bytes(model_content))
    
    # TFLM uses different API
    INPUT_INDEX = 0
    OUTPUT_INDEX = 0
    input_details = interpreter.get_input_details(INPUT_INDEX)
    output_details = interpreter.get_output_details(OUTPUT_INDEX)
    
    predictions = []
    
    for img in images:
        test_image = img[np.newaxis, ...]
        
        if input_details['dtype'] == np.uint8:
            quant_params = input_details.get('quantization_parameters', {})
            if 'scales' in quant_params and 'zero_points' in quant_params:
                input_scale = quant_params['scales'][0]
                input_zero_point = quant_params['zero_points'][0]
                test_image = test_image / input_scale + input_zero_point
                test_image = test_image.astype(np.uint8)
        
        interpreter.set_input(test_image, INPUT_INDEX)
        interpreter.invoke()
        
        output = interpreter.get_output(OUTPUT_INDEX)[0]
        
        if output_details['dtype'] == np.uint8:
            quant_params = output_details.get('quantization_parameters', {})
            if 'scales' in quant_params and 'zero_points' in quant_params:
                output_scale = quant_params['scales'][0]
                output_zero_point = quant_params['zero_points'][0]
                output = (output.astype(np.float32) - output_zero_point) * output_scale
        
        predictions.append(np.argmax(output))
    
    return predictions

# Get sample images
sample_indices = np.random.choice(len(test_images), 10, replace=False)
sample_images = test_images[sample_indices]
sample_labels = test_labels[sample_indices]

# Get predictions from original and compressed models
original_preds = get_predictions(tflite_model, sample_images)
compressed_preds = get_predictions(compressed_model, sample_images)

# Visualize
fig, axes = plt.subplots(2, 5, figsize=(15, 8))
axes = axes.ravel()

for i in range(10):
    axes[i].imshow(sample_images[i].squeeze(), cmap='gray')
    axes[i].set_title(f'True: {sample_labels[i]}\nOrig: {original_preds[i]}, Compressed: {compressed_preds[i]}',
                      color='green' if original_preds[i] == compressed_preds[i] == sample_labels[i] else 'red')
    axes[i].axis('off')

plt.suptitle('Original vs TFLM Compressed Model Predictions\n(Green = all correct, Red = at least one incorrect)')
plt.tight_layout()
plt.show()
```

## Summary and Key Takeaways

Congratulations! You've successfully compressed a neural network for microcontroller
deployment using TFLM's compression techniques.

### What We Accomplished

Through this tutorial, you learned the complete compression pipeline:

1. **Baseline Model Creation**: We built a simple CNN for MNIST classification,
   achieving good accuracy while keeping the architecture microcontroller-friendly.

2. **Selective Weight Clustering**: We applied clustering to only the second Conv2D
   layer, demonstrating how to balance compression and accuracy by targeting specific
   layers. The 16 clusters prepared this layer for 4-bit compression.

3. **Quantization**: We converted the model from float32 to INT8, achieving 4x size
   reduction while maintaining inference accuracy.

4. **TFLM LUT Compression**: We applied Look-Up Table compression to the clustered
   layers, storing weights as 4-bit indices into tables of 16 values, achieving
   additional 2x compression for those layers.

### Key Insights

- **Clustering is Essential**: Without clustering, weights are too diverse to compress
  effectively. Clustering creates the redundancy that compression algorithms exploit.

- **Selective Compression Works**: You don't need to compress every layer. Target the
  largest layers or those least sensitive to compression for optimal results.

- **Compression is a Tradeoff**: We achieved significant size reduction (6-8x
  overall) with minimal accuracy loss. Your specific tradeoff will depend on your
  application's requirements.

### Practical Considerations for Production

1. **Operator Support**: Currently, not all TFLM operators support compression. Check
   the documentation for your target operators and plan accordingly.

2. **Compression Planning**: Before training, consider which layers you'll compress.
   You might design your architecture with compression in mind.

3. **Fine-tuning Matters**: Fine-tune after clustering to let the model adapt to the
   constrained weight space.

4. **Test on Target Hardware**: While TFLM's Python interpreter provides accurate
   results, validate inference speed on your actual microcontroller.

### Next Steps

- Experiment with different clustering configurations (4, 8, or 32 clusters)
- Try compressing different combinations of layers

### Resources

- [TensorFlow Model Optimization Toolkit](https://www.tensorflow.org/model_optimization)
- [TFLM Compression Documentation](https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/compression)
- [Running Compressed Models on Simulators and Hardware](https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/docs/compression.md)
- [TensorFlow Lite for Microcontrollers](https://www.tensorflow.org/lite/microcontrollers)
