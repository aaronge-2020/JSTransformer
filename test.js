import {
  positionalEncoding,
  PositionalEmbedding,
  MultiHeadAttention,
  GlobalSelfAttention,
  CrossAttention,
  CausalSelfAttention,
  Encoder,
  DecoderLayer,
  Decoder,
  Transformer
} from "./transformer.js";

// Example usage
// const model = new SelfAttention({k:128, heads: 4});
// const input = tf.ones([32, 10, 128]); // Batch size: 32, Sequence length: 10, Embedding size: 128
// const output = model.call(input);
// output.print();
// positionalEncoding(10, 512).print()

// Test the PositionalEmbedding layer
// const vocabSize = 5000;
// const dModel = 512;
// const inputSequence = tf.tensor2d([
//   [1, 2, 3, 4, 0, 0, 0, 0],
//   [5, 6, 7, 0, 9, 1, 1, 2],
// ]);

// const layer = new PositionalEmbedding(vocabSize, dModel);
// const output = layer.call(inputSequence);

// output.print();

// test the custom layer MHA
// Create MultiHeadAttention instance
//   const multi = new MultiHeadAttention(512, 2);

//   // Create random tensor of shape [1, 60, 512]
//   const y = tf.randomUniform([1, 60, 512]);

//   // Call the MultiHeadAttention instance
//   const [out, attn] = multi.call(y, y, y, null);

//   // Print the shape of the output and attention tensors
//   console.log(`out shape: ${out.shape}`);
//   console.log(`attn shape: ${attn.shape}`);

//  test the custom layer SelfAttention
// Create SelfAttention instance
// const self = new CrossAttention(512, 2);

// // Create input tensors
// const x = tf.randomNormal([1, 60, 512]);
// const context = tf.randomNormal([1, 60, 512]);

// // Call the CrossAttention instance
// const output = self.call([x, context]);

// // Print the shape of the output tensor
// console.log('Output shape:', output.shape);

// //  test the custom layer Causal Self Attention
// const causalSelfAttention = new CausalSelfAttention(128, 4, {});

// // Create a mock input tensor [batch_size, sequence_length, d_model]
// // For simplicity, let's assume batch_size=1, sequence_length=5, d_model=128
// const inputTensor = tf.randomNormal([1, 5, 128]);

// // Run the forward pass
// const outputTensor = causalSelfAttention.call(inputTensor);

// // Print the output shape
// console.log('Output Tensor Shape:', outputTensor.shape);

// // Check output shape [1, 5, 128]
// if (outputTensor.shape.toString() !== '1,5,128') {
//   console.log('Test failed: Unexpected output shape.');
// }

// // Check causality (This is a rough check; for rigorous tests, you'd need known inputs and outputs)
// const outputArray = await outputTensor.array();
// const inputArray = await inputTensor.array();

// // Create an instance of the Encoder
// const encoder = new Encoder(2, 512, 8, 2048, 10000, 0.1);

// // Create some mock data (batch_size: 3, seq_len: 4)
// const mockData = tf.tensor([
//   [1, 2, 3, 4],
//   [5, 6, 7, 8],
//   [9, 10, 11, 12],
// ]);

// // Test the Encoder
// const output = encoder.call(mockData);

// // Print the output shape (should be [3, 4, 512] if everything is set up correctly)
// output.print();


// // Test the decoder layer
// // Initialize the DecoderLayer
// const d_model = 512;
// const num_heads = 8;
// const dff = 2048;
// const dropout_rate = 0.1;

// const decoderLayer = new DecoderLayer(d_model, num_heads, dff, dropout_rate);

// // Create some dummy data to pass through the layer
// const batchSize = 64;
// const seqLen = 50;

// const x = tf.randomNormal([batchSize, seqLen, d_model]);  // Your input tensor
// const context = tf.randomNormal([batchSize, seqLen, d_model]);  // Your context tensor

// // Run the data through the layer
// const output = decoderLayer.call([x, context]);

// // Inspect the output
// output.print();

// // Instantiate the decoder
// const sampleDecoder = new Decoder(1, 128, 4, 1028, 1000);

// // Create some mock data for 'x' and 'context'
// // Assuming 'x' and 'context' are 3D tensors.
// // Replace these lines with your actual tensors.
// const en = tf.randomNormal([64, 50, 512]); // Mock shape [batch_size, sequence_length, d_model]
// const pt_emb = tf.randomNormal([64, 50, 512]); // Mock shape [batch_size, sequence_length, d_model]

// // Test the Decoder
// const output = sampleDecoder.call([en, pt_emb]);

// // Print the shapes
// en.shape.print();
// pt_emb.shape.print();
// output.shape.print();

// Test the Transformer
const num_layers = 4
const d_model = 128
const dff = 512
const num_heads = 8
const dropout_rate = 0.1

const transformer = new Transformer(
  num_layers,
  d_model,
  num_heads,
  dff,
  1000,
  1000,
  0.1)

// output = transformer((pt, en))

// print(en.shape)
// print(pt.shape)
// print(output.shape)