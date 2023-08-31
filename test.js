import {
  positionalEncoding,
  PositionalEmbedding,
  MultiHeadAttention,
  GlobalSelfAttention,
  CrossAttention
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
const self = new CrossAttention(512, 2);

// Create input tensors
const x = tf.randomNormal([1, 60, 512]);
const context = tf.randomNormal([1, 60, 512]);

// Call the CrossAttention instance
const output = self.call([x, context]);

// Print the shape of the output tensor
console.log('Output shape:', output.shape);